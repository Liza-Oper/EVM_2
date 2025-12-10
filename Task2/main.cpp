#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

double f(double x) {
    return sin(x);
}

struct Spline {
    vector<double> x, y, a, b, c, d;  };

Spline buildSpline(const vector<double>& x, const vector<double>& y) {
    int n = x.size();
    vector<double> h(n-1);
    for (int i = 0; i < n-1; i++)
        h[i] = x[i+1] - x[i];

    vector<double> alpha(n);
    for (int i = 1; i < n-1; i++)
        alpha[i] = (3.0/h[i])*(y[i+1]-y[i]) - (3.0/h[i-1])*(y[i]-y[i-1]);

    vector<double> l(n), mu(n), z(n);
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;
    for (int i = 1; i < n-1; i++) {
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1];
        mu[i] = h[i] / l[i];
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / l[i];
    }

    vector<double> c(n), b(n), d(n);
    l[n-1] = 1;
    z[n-1] = 0;
    c[n-1] = 0;

    for (int j = n-2; j >= 0; j--) {
        c[j] = z[j] - mu[j]*c[j+1];
        b[j] = (y[j+1] - y[j])/h[j] - h[j]*(c[j+1] + 2*c[j])/3.0;
        d[j] = (c[j+1] - c[j]) / (3.0*h[j]);
    }

    Spline spl;
    spl.x = x;
    spl.y = y;
    spl.a = y;
    spl.b = b;
    spl.c = c;
    spl.d = d;
    return spl;}

double S(const Spline& spl, double X) {
    int n = spl.x.size();
    int i = 0;
    while (i < n-2 && X > spl.x[i+1]) i++;
    double dx = X - spl.x[i];
    return spl.a[i] + spl.b[i] * dx+ spl.c[i] * dx*dx / 2.0 + spl.d[i] * dx*dx*dx; }

int main() {
    vector<int> Ns = {10, 50, 100, 1000};

    cout << fixed << setprecision(10);

    for (int N : Ns) {
        cout << "\n==============================\n";
        cout << "\n==============================\n";
        cout << "           N = " << N << "\n";
        cout << "==============================\n";
        cout << "\n==============================\n";

        vector<double> x(N), y(N);
        for (int i = 0; i < N; i++) {
            x[i] = double(i) / (N - 1);
            y[i] = f(x[i]);
        }

        Spline spl = buildSpline(x, y);
        double maxErr = 0;
        for (int i = 0; i < 2000; i++) {
            double xx = i / 1999.0;
            double real_val = f(xx);
            double spline_val = S(spl, xx);
            maxErr = max(maxErr, fabs(real_val - spline_val));  }

        cout << "Max interpolation error: " << maxErr << "\n";

        // for (int i = 0; i < 1000; i++) {
        for (int i = 0; i < 10; i++) {
            double xx = i / 1999.0;
            double spline_val = S(spl, xx);
            double real_val = f(xx);
            cout << xx << " " << real_val << " " << spline_val << "\n";
        }
    }
    return 0; }
