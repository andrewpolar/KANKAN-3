#pragma once
#include <memory>
#include <vector>

class Helper
{
public:
    static double Pearson(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, int len) {
        double xmean = 0.0;
        double ymean = 0.0;
        for (int i = 0; i < len; ++i) {
            xmean += x[i];
            ymean += y[i];
        }
        xmean /= len;
        ymean /= len;

        double covariance = 0.0;
        for (int i = 0; i < len; ++i) {
            covariance += (x[i] - xmean) * (y[i] - ymean);
        }

        double stdX = 0.0;
        double stdY = 0.0;
        for (int i = 0; i < len; ++i) {
            stdX += (x[i] - xmean) * (x[i] - xmean);
            stdY += (y[i] - ymean) * (y[i] - ymean);
        }
        stdX = sqrt(stdX);
        stdY = sqrt(stdY);
        return covariance / stdX / stdY;
    }
    static void ShowMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%5.3f ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    static void ShowVector(std::unique_ptr<double[]>& ptr, int N) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            printf("%5.2f ", ptr[i]);
            if (++cnt >= 10) {
                printf("\n");
                cnt = 0;
            }
        }
    }
    static void SwapRows(std::unique_ptr<double[]>& row1, std::unique_ptr<double[]>& row2, int cols) {
        auto ptr = std::make_unique<double[]>(cols);
        for (int i = 0; i < cols; ++i) {
            ptr[i] = row1[i];
        }
        for (int i = 0; i < cols; ++i) {
            row1[i] = row2[i];
        }
        for (int i = 0; i < cols; ++i) {
            row2[i] = ptr[i];
        }
    }
    static void SwapScalars(double& x1, double& x2) {
        double buff = x1;
        x1 = x2;
        x2 = buff;
    }
    static void Shuffle(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, std::unique_ptr<double[]>& vector, int rows, int cols) {
        for (int i = 0; i < 2 * rows; ++i) {
            int n1 = rand() % rows;
            int n2 = rand() % rows;
            SwapRows(matrix[n1], matrix[n2], cols);
            SwapScalars(vector[n1], vector[n2]);
        }
    }
    static void FindMinMaxMatrix(std::vector<double>& xmin, std::vector<double>& xmax,
        std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int nRows, int nCols) {
        for (int i = 0; i < nCols; ++i) {
            xmin.push_back(DBL_MAX);
            xmax.push_back(-DBL_MAX);
        }
        for (int i = 0; i < nRows; ++i) {
            for (int j = 0; j < nCols; ++j) {
                if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
                if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
            }
        }
    }

    static void FindMinMax(std::vector<double>& xmin, std::vector<double>& xmax,
        double& targetMin, double& targetMax,
        std::unique_ptr<std::unique_ptr<double[]>[]>& matrix,
        std::unique_ptr<double[]>& target, int nRows, int nCols) {

        for (int i = 0; i < nCols; ++i) {
            xmin.push_back(DBL_MAX);
            xmax.push_back(-DBL_MAX);
        }
        for (int i = 0; i < nRows; ++i) {
            for (int j = 0; j < nCols; ++j) {
                if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
                if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
            }
        }
        targetMin = DBL_MAX;
        targetMax = -DBL_MAX;
        for (int j = 0; j < nRows; ++j) {
            if (target[j] < targetMin) targetMin = target[j];
            if (target[j] > targetMax) targetMax = target[j];
        }
    }

    static double Min(const std::unique_ptr<double[]>& x, int N) {
        double min = x[0];
        for (int i = 1; i < N; ++i) {
            if (x[i] < min) min = x[i];
        }
        return min;
    }

    static double Max(const std::unique_ptr<double[]>& x, int N) {
        double max = x[0];
        for (int i = 1; i < N; ++i) {
            if (x[i] > max) max = x[i];
        }
        return max;
    }

    static double MinV(const std::vector<double>& x) {
        double min = x[0];
        for (int i = 1; i < x.size(); ++i) {
            if (x[i] < min) min = x[i];
        }
        return min;
    }

    static double MaxV(const std::vector<double>& x) {
        double max = x[0];
        for (int i = 1; i < x.size(); ++i) {
            if (x[i] > max) max = x[i];
        }
        return max;
    }
};
