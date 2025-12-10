#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace std;

long double PI = 3.14159265;

double f(double x) {
    return x * (1 - x) * log(1 - x);
}


vector<complex<double>> DFT(const vector<double>& a) {
    int N = a.size();
    vector<complex<double>> F(N);
    for (int k = 0; k < N; k++) {
        complex<double> sum = 0;
        for (int n = 0; n < N; n++) {
            double angle = -2 * PI * k * n / N;
            sum += a[n] * complex<double>(cos(angle), sin(angle));
        }
        F[k] = sum;
    }
    return F;
}

vector<double> Inver_DFT(const vector<complex<double>>& F) {
    int N = F.size();
    vector<double> a(N);
    for (int n = 0; n < N; n++) {
        complex<double> sum = 0;
        for (int k = 0; k < N; k++) {
            double angle = 2 * PI * k * n / N;
            sum += F[k] * complex<double>(cos(angle), sin(angle));
        }
        a[n] = sum.real() / N;
    }
    return a;
}

int main() {
    vector<int> Ns = {10, 50, 100, 1000};

    for (int N : Ns) {
        cout << "N = " << N << endl;
        vector<double> x(N), fvals(N);
        for (int i = 0; i < N; i++) {
            x[i] = (double)i / N;
            if (x[i] == 1) x[i] = 1 - 1e-12;
            fvals[i] = f(x[i]);
        }

        auto F = DFT(fvals);
        auto f_rec = Inver_DFT(F);

        double err = 0;
        for (int i = 0; i < N; i++)
            err += fabs(fvals[i] - f_rec[i]);

        cout << "Error = " << err << "\n\n";
    }
    return 0;}
