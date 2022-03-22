#ifndef __DISCRETE_ELASTIC_RODS_H__
#define __DISCRETE_ELASTIC_RODS_H__
using namespace std;
using namespace Eigen;

class DiscreteElasticRods {
public:
    int nv;
    VectorXd x;
    vector<bool> is_fixed;
    VectorXd v;
    VectorXd e;
    VectorXd length;
    VectorXd length_temp;
    VectorXd l_temp;
    MatrixX3d kb;
    MatrixX2d kappa_temp;

    //material frame
    MatrixX3d m1;
    MatrixX3d m2;
    MatrixX3d m1_temp;
    MatrixX3d m2_temp;
    VectorXd theta;

    //visualization
    double bending_energy = 0.0;
    vector<Vector3d> visual_bending_force;

    DiscreteElasticRods();

    void initSimulation(int nv_, VectorXd x_, VectorXd theta_, vector<bool> is_fixed_);

    void startSimulate();

    void updateCenterlinePosition(void);

    void computeGH(VectorXd& gradient, SparseMatrix<double>& hessian, MatrixX3d& t);

    void centerlineVelocity(VectorXd& gradient);

    void updateEdge();

    MatrixX3d unitTangents(VectorXd& x_);

    void curvatureBinormal(MatrixX3d t);

    void materialCurvature(MatrixX2d& kappa);

    double bendingForce(VectorXd& gradient, vector<Triplet<double>>& hessian, MatrixX3d& t, VectorXd& bending_force);
};

#endif
