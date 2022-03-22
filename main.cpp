#include <igl/opengl/glfw/Viewer.h>
#include "DiscreteElasticRods.h"
using namespace std;
using namespace Eigen;

//create a line for visualization
static void initVisualization(int nv, VectorXd x_, vector< Vector3d>* visual_nodes)
{
    for (int i = 0; i<nv; i++) {
      Vector3d node(x_(3*i), x_(3*i+1), x_(3*i+2));
      visual_nodes->push_back(node);
    }
}

//update the line's position after applying energy
static void updateVisualization(int nv, VectorXd x_, vector< Vector3d>* visual_nodes)
{
    for (int i = 0; i<nv; i++) {
      Vector3d node(x_(3*i), x_(3*i+1), x_(3*i+2));
      visual_nodes->at(i)= node;
    }
}

int main(int argc, char *argv[])
{ 
  DiscreteElasticRods discrete_elastic_rods;

  //create nodes for visualization
  vector< Vector3d> visual_nodes;

  //initiate basic values
  int nv = 0;
  VectorXd x_;
  vector<bool> is_fixed;
  VectorXd theta;

  nv = 10; //we set 10 nodes as our default value. when setting to nv=5, errors exist
  x_.resize(nv*3);
  x_.setZero();
  theta.resize(nv-1);
  theta.setZero();

  //decide to stick the first two nodes still and compute rest of the nodes
  for (int i = 0; i<nv; i++) {
    x_(3*i) = (-(nv >> 1)+i);
    is_fixed.push_back(false);
  }
  for (int i = 0; i<2; i++) {
    is_fixed[i] = true;
  }

  discrete_elastic_rods.initSimulation(nv, x_, theta, is_fixed);
  initVisualization(nv, x_, &visual_nodes);

  //convert the format of the matrix (https://libigl.github.io/tutorial/)
  Map<MatrixXd> sensor_input = Map<MatrixXd>(visual_nodes[0].data(),3,visual_nodes.size());
  MatrixXd Vec = sensor_input.transpose();
  cout << "initial vector: \n" << Vec << endl;

  //const MatrixXi Edg = (MatrixXi(8,3)<< 1,2,3, 2,3,4, 3,4,5, 4,5,6, 5,6,7, 6,7,8, 7,8,9, 8,9,10).finished().array()-1;
  const MatrixXi Edg = (MatrixXi(9,3)<< 1,1,2, 2,2,3, 3,3,4, 4,4,5, 5,5,6, 6,6,7, 7,7,8, 8,8,9, 9,9,10).finished().array()-1;

  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_face_based(true);
  viewer.core().is_animating = true;
  MatrixXd P = (Eigen::MatrixXd(1,3)<<1.42,0,0).finished();

  //function will be called before every draw
  viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer & )->bool
  {
    //create orbiting animation
    P = (1.42*(P+ RowVector3d(P(1),-P(0),P(2))*0.1).normalized()).eval();
    //update point location. no .clear() necessary
    viewer.data().set_points(P, RowVector3d(1,1,1));

    //update the move i calculations per time so the simulation moves more obviously
    for (int i = 0; i < 1; i++)
    {
      discrete_elastic_rods.startSimulate();
    }
    
    updateVisualization(discrete_elastic_rods.nv, discrete_elastic_rods.x, &visual_nodes);
    sensor_input =  Map<MatrixXd>(visual_nodes[0].data(),3,visual_nodes.size());
    Vec = sensor_input.transpose();
    cout << "updated vector: \n" << Vec << endl;
    viewer.data().set_mesh(Vec, Edg);
    return 0;
  };

  //plot the mesh and update to the window
  viewer.data().set_mesh(Vec, Edg);
  viewer.launch();    
}

DiscreteElasticRods::DiscreteElasticRods() { }

void DiscreteElasticRods::initSimulation(int nv_, VectorXd x_, VectorXd theta_, vector<bool> is_fixed_)
{
    nv = nv_;
    x = move(x_);
    is_fixed = move(is_fixed_);
    v.resize(nv*3);
    v.setZero();

    e.resize((nv-1)*3);
    e.setZero();
    length_temp.resize(nv-1);
    length_temp.setZero();
    length.resize(nv-1);
    length.setZero();
    updateEdge(); //update each segment of edges after calculation then normalize edges

    for (int i = 0; i<nv-1; i++) {
        length_temp(i) = length(i);
    }
    MatrixX3d t = unitTangents(x); //in paper 4.2, we consider adapted frames, where ti = ei/|ei| is the unit tangent vector per edge

    //set up reference frame
    theta = move(theta_);
    m1_temp.resize(nv-1, 3);
    m2_temp.resize(nv-1, 3);

    for (int i = 0; i<nv-1; i++) {
        Vector3d t_i = t.row(i).transpose();
        m1_temp.row(i) =  RowVector3d(-t_i(1), t_i(0), 0);
        Vector3d m1_i = m1_temp.row(i).transpose();
        m2_temp.row(i) = (m1_i.cross(t_i)).transpose();
    }
    m1.resize(nv-1, 3);
    m1 = m1_temp;
    m2.resize(nv-1, 3);
    m2 = m2_temp;

    kb.resize(nv-2, 3);
    curvatureBinormal(t); //equation(1)
    materialCurvature(kappa_temp); //equation(2)
    
    //in paper 4.2, define Di as the Voronoi region associated to each vertex, having length |Di| = li/2,where li = |ei−1|+|ei|
    l_temp.resize(nv-2);
    for (int i = 0; i<nv-2; i++) {
        l_temp(i) = (length_temp(i)+length_temp(i+1))/2;
    }
    
    visual_bending_force.resize(nv);
}


void DiscreteElasticRods::startSimulate()
{
    updateCenterlinePosition();
    MatrixX3d t = unitTangents(x);

    updateEdge();
    curvatureBinormal(t);

    VectorXd gradient;
    SparseMatrix<double> hessian;
    computeGH(gradient, hessian, t); //equation(7)
    centerlineVelocity(gradient);
}

void DiscreteElasticRods::updateCenterlinePosition(void)
{
    const double h = 0.005; //time step
    for (int i = 0; i<nv; i++) {
        if (!is_fixed[i]) x.segment<3>(3*i) += h*v.segment<3>(3*i);
    }
}

void DiscreteElasticRods::computeGH(VectorXd& gradient, SparseMatrix<double>& hessian, MatrixX3d& t)
{
    VectorXd bending_force;
    bending_force.resize(3*nv);
    bending_force.setZero();

    //to initialize, set zero to gradient and hessian
    int ndof = 3*nv+nv-1;
    VectorXd gradient_temp(ndof);
    gradient_temp.setZero();
    SparseMatrix<double> hessian_temp(ndof, ndof);
    tie(gradient, hessian) = make_tuple(gradient_temp, hessian_temp);

    vector< Triplet<double>> hessian_triplets;

    //since we implement an unextensible rod, there's no need to add stretching energy
    bending_energy = bendingForce(gradient, hessian_triplets, t, bending_force);

    //adding gravity into calculation
    const double g = -9.8;
    for (int i = 0; i<nv; i++) {
        gradient(3*i+1) -= g;
    }

    hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());

    //build visualization after adding bending force and gravity
    for (int i = 0; i<nv; i++) {
        visual_bending_force[i] = bending_force.segment<3>(3*i);
    }
}

void DiscreteElasticRods::centerlineVelocity(VectorXd& gradient)
{
    const double h = 0.005; //time
    for (int i = 0; i<nv; i++) {
        v.segment<3>(3*i) -= h*gradient.segment<3>(3*i);
    }
}

void DiscreteElasticRods::updateEdge()
{
    for (int i = 0; i<nv-1; i++) {
        e.segment<3>(3*i) = x.segment<3>(3*(i+1))-x.segment<3>(3*i);
        //cout << e << endl;
    }
    for (int i = 0; i<nv-1; i++) {
        length(i) = e.segment<3>(3*i).norm();
        //cout << length(i) << endl;
    }
}

MatrixX3d DiscreteElasticRods::unitTangents(VectorXd& x_)
{
    MatrixX3d unit_tangents(nv-1, 3);
    for (int i = 0; i<nv-1; i++) {
        unit_tangents.row(i) = ((x_.segment<3>(3*(i+1))-x_.segment<3>(3*i)).normalized()).transpose(); //assign frames to edges, rather than to vertices
        //cout << unit_tangents.row(i) << endl;
    }
    return unit_tangents;
}

void DiscreteElasticRods::curvatureBinormal(MatrixX3d t)
{
    for (int i = 0; i<nv-2; i++) {
        Vector3d t_i = t.row(i).transpose();
        //cout << t_i << endl;
        Vector3d t_ip1 = t.row(i+1).transpose();
        //cout << t_ip1 << endl;
        Vector3d kb_i = 2*(t_i).cross(t_ip1)/(1+t_i.dot(t_ip1));
        kb.row(i) = kb_i.transpose();
    }
}

void DiscreteElasticRods::materialCurvature(MatrixX2d& kappa)
{
    kappa.resize(nv-2, 2);
    for (int i = 0; i<nv-2; i++) {
        VectorXd kb_i = kb.row(i).transpose();
        //kappa
        kappa(i, 0) = kb_i.dot((m2.row(i)+m2.row(i+1)).transpose())/2;
        kappa(i, 1) = -kb_i.dot((m1.row(i)+m1.row(i+1)).transpose())/2;
    }
}

double DiscreteElasticRods::bendingForce(VectorXd& gradient, vector<Triplet<double>>& hessian, MatrixX3d& t, VectorXd& bending_force)
{
    vector< Triplet<double>> hess_bending_triplets;
    const double r = 0.1; //segment radius
    double E_b = 0.0;
    MatrixX2d kappa;
    materialCurvature(kappa);

    for (int i = 1; i<nv-1; i++) {
        const double area = M_PI*r*r; //area = pi*r*r
        const double B_half = 100000000*area*r*r/4; //bending modulus = 1000000
        const double l_i = l_temp(i-1);
        const double kappa1_i = kappa(i-1, 0);
        const double kappa2_i = kappa(i-1, 1);

        //compute bending energy
        E_b += (B_half*(kappa1_i-kappa_temp(i-1, 0)) * (kappa1_i-kappa_temp(i-1, 0)) + B_half*(kappa2_i-kappa_temp(i-1, 1)) * (kappa2_i-kappa_temp(i-1, 1)))/l_i;

        //compute bending force gradient
        VectorXd gradient_kappa1_i(11); //3*3(x DOF) + 2(theta DOF)
        VectorXd gradient_kappa2_i(11); //3*3(x DOF) + 2(theta DOF)
        const double temp = 1+t.row(i-1).dot(t.row(i));
        Vector3d kb_i = kb.row(i-1).transpose();
        Vector3d m1_im1 = m1.row(i-1).transpose();
        Vector3d m1_i = m1.row(i).transpose();
        Vector3d m1_tilde = (m1_i+m1_im1)/temp;
        Vector3d m2_im1 = m2.row(i-1).transpose();
        Vector3d m2_i = m2.row(i).transpose();
        Vector3d m2_tilde = (m2_i+m2_im1)/temp;
        Vector3d t_im1 = t.row(i-1).transpose();
        Vector3d t_i = t.row(i).transpose();
        Vector3d t_tilde = (t_i+t_im1)/temp;

        Matrix3d de_dx_im1 = - Matrix3d::Identity();
        Matrix3d de_dx_i =  Matrix3d::Identity();

        Vector3d dkappa1_i_de_im1 = m2_tilde.cross(-t_i/length_temp(i-1))-kappa1_i*t_tilde/length_temp(i-1);
        Vector3d dkappa1_i_de_i = m2_tilde.cross(t_im1/length_temp(i))-kappa1_i*t_tilde/length_temp(i);
        Vector3d dkappa2_i_de_im1 = m1_tilde.cross(t_i/length_temp(i-1))-kappa2_i*t_tilde/length_temp(i-1);
        Vector3d dkappa2_i_de_i = m1_tilde.cross(-t_im1/length_temp(i))-kappa2_i*t_tilde/length_temp(i);

        //for x
        gradient_kappa1_i.segment<3>(0) = dkappa1_i_de_im1.transpose()*de_dx_im1;
        gradient_kappa1_i.segment<3>(3) = dkappa1_i_de_im1.transpose()*de_dx_i+dkappa1_i_de_i.transpose()*de_dx_im1;
        gradient_kappa1_i.segment<3>(6) = dkappa1_i_de_i.transpose()*de_dx_i;
        gradient_kappa2_i.segment<3>(0) = dkappa2_i_de_im1.transpose()*de_dx_im1;
        gradient_kappa2_i.segment<3>(3) = dkappa2_i_de_im1.transpose()*de_dx_i+dkappa2_i_de_i.transpose()*de_dx_im1;
        gradient_kappa2_i.segment<3>(6) = dkappa2_i_de_i.transpose()*de_dx_i;
        //for theta
        gradient_kappa1_i(9) = kb_i.dot(m1_im1)/2;
        gradient_kappa1_i(10) = kb_i.dot(m1_i)/2;
        gradient_kappa2_i(9) = -kb_i.dot(m2_im1)/2;
        gradient_kappa2_i(10) = -kb_i.dot(m2_i)/2;
 
        //update dx
        Matrix<double, 9, 1> gradient_dx = (B_half*(kappa1_i-kappa_temp(i-1, 0))*gradient_kappa1_i.segment<9>(0) + B_half*(kappa2_i-kappa_temp(i-1, 1))*gradient_kappa2_i.segment<9>(0))/l_i;
        gradient.segment<9>(3*(i-1)) += gradient_dx;
        bending_force.segment<9>(3*(i-1)) -= gradient_dx.segment<9>(0);
        //update dtheta
        gradient.segment<2>(3*nv+i-1) += (B_half*(kappa1_i-kappa_temp(i-1, 0))*gradient_kappa1_i.segment<2>(9) + B_half*(kappa2_i-kappa_temp(i-1, 1))*gradient_kappa2_i.segment<2>(9))/l_i;
    }
    E_b /= 2;
    //check and compute bending force hessian
    hessian.insert(hessian.end(), hess_bending_triplets.begin(), hess_bending_triplets.end());

    return E_b;
}
