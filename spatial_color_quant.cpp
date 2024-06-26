/* Copyright (c) 2006 Derrick Coetzee

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

using namespace std;

template <typename T, size_t length>
class vector_fixed
{
public:
    vector_fixed()
    {
        data.fill(T{});
    }

    vector_fixed(const vector_fixed<T, length>& rhs)
    {
        data = rhs.data;
    }

    vector_fixed(const std::vector<T>& rhs)
    {
        assert(rhs.size() == length);
        // The original code does a copy_n of length elements, in case that assert gets triggered.
        std::copy(rhs.data.cbegin(), rhs.data.cend(), data.begin());
    }

    T& operator()(size_t i) {
        return data.at(i);
    }

    const T& operator()(size_t i) const
    {
        return data.at(i);
    }

    constexpr size_t get_length() const { return length; }

    T norm_squared() const {
        // Performance could improve if this was switched to transform_reduce.
        // But then this would also become non-deterministic.
        return std::inner_product(data.cbegin(), data.cend(), data.cbegin(), T{});
    }

    vector_fixed<T, length>& operator=(const vector_fixed<T, length>& rhs)
    {
        data = rhs.data;
        return *this;
    }

    vector_fixed<T, length> direct_product(const vector_fixed<T, length>& rhs) const {
        vector_fixed<T, length> result;
        std::transform(data.cbegin(), data.cend(), rhs.data.cbegin(), result.data.begin(), std::multiplies<>());
        return result;
    }

    double dot_product(vector_fixed<T, length> rhs) const {
        return std::inner_product(data.cbegin(), data.cend(), rhs.data.cbegin(), T{});
    }

    vector_fixed<T, length>& operator+=(const vector_fixed<T, length> rhs) {
        std::transform(data.cbegin(), data.cend(), rhs.data.cbegin(), data.begin(), std::plus<>());
        return *this;
    }

    vector_fixed<T, length> operator+(vector_fixed<T, length> rhs) const {
        vector_fixed<T, length> result(*this);
        result += rhs;
        return result;
    }

    vector_fixed<T, length>& operator-=(vector_fixed<T, length> rhs) {
        std::transform(data.cbegin(), data.cend(), rhs.data.cbegin(), data.begin(), std::minus<>());
        return *this;
    }

    vector_fixed<T, length> operator-(vector_fixed<T, length> rhs) const {
        vector_fixed<T, length> result(*this);
        result -= rhs;
        return result;
    }

    vector_fixed<T, length>& operator*=(T scalar) {
        std::transform(data.cbegin(), data.cend(), data.begin(), [scalar](const T val) { return val * scalar; });
        return *this;
    }

    vector_fixed<T, length> operator*(T scalar) const {
        vector_fixed<T, length> result(*this);
        result *= scalar;
        return result;
    }

private:
    std::array<T, length> data;
};

template <typename T, size_t length>
vector_fixed<T, length> operator*(T scalar, vector_fixed<T, length> vec) {
    return vec*scalar;
}


template <typename T, size_t length>
ostream& operator<<(ostream& out, vector_fixed<T, length> vec) {
    out << "(";
    int i;
    for (i=0; i<length - 1; i++) {
        out << vec(i) << ", ";
    }
    out << vec(i) << ")";
    return out;
}

template <typename T>
class array2d
{
public:
    array2d(size_t w, size_t h)
    : data{make_unique<T[]>(w * h)},
      width{w},
      height{h}
    {
    }

    array2d(const array2d<T>& rhs)
    : data{std::make_unique<T[]>(rhs.width * rhs.height)},
      width{rhs.width},
      height{rhs.height}
    {
        for(size_t i=0; i<width; i++) {
            for(size_t j=0; j<height; j++) {
                (*this)(i,j) = rhs.data[j*width + i];
            }
        }
    }
    
    array2d(array2d<T>&& rhs)
    : data{std::move(rhs.data)},
      width{rhs.width},
      height{rhs.height}
    {
        rhs.width = 0;
        rhs.height = 0;
    }
    

    ~array2d()
    {
    }

    T& operator()(size_t col, size_t row)
    {
        assert(col < width);
        assert(row < height);
        size_t index = row*width + col;
        return data[index];
    }
    
    const T& operator()(size_t col, size_t row) const
    {
        assert(col < width);
        assert(row < height);
        size_t index = row*width + col;
        return data[index];
    }

    size_t get_width() const { return width; }
    size_t get_height() const { return height; }

    array2d<T>& operator*=(T scalar) {
        std::transform(data.get(), data.get() + (width * height), data.get(), [scalar](const T val){ return val * scalar; });
        return *this;
    }

    array2d<T> operator*(T scalar) const {
        array2d<T> result(*this);
        result *= scalar;
        return result;
    }

    // Matrix-vector multiplication... literally
    std::vector<T> operator*(const std::vector<T>& vec) const {
        std::vector<T> result;
        assert(vec.size() == width);
        result.reserve(vec.size());

        for (size_t row = 0; row < height; row++) {
            // T sum = 0;
            // for(int col=0; col<get_width(); col++) {
            //     sum += (*this)(col,row) * vec[col];
            // }
            result.push_back(std::inner_product(&(*this)(0, row), &(*this)(0, row) + width, vec.cbegin(), T{}));
        }
        return result;
    }

    array2d<T>& multiply_row_scalar(size_t row, double mult) {
        T* row_iter = &(*this)(0, row);
        std::transform(row_iter, row_iter + width, row_iter, [mult](const T val){ return val * mult; });
        return *this;
    }

    array2d<T>& add_row_multiple(size_t from_row, size_t to_row, double mult) {
        const T* from_row_iter = &(*this)(0, from_row);
        T* to_row_iter = &(*this)(0, to_row);
        std::transform(from_row_iter, from_row_iter + width, to_row_iter, to_row_iter, [mult](const T val1, const T val2){ return val1 * mult + val2; });
        return *this;
    }

    // We use simple Gaussian elimination - perf doesn't matter since
    // the matrices will be K x K, where K = number of palette entries.
    array2d<T> matrix_inverse() {
        array2d<T> result(width, height);
        array2d<T>& a = *this;

        // Set result to identity matrix
        std::fill(result.data.get(), result.data.get() + (width * height), 0.0);
        for(size_t i=0; i<width; i++) {
            result(i,i) = 1;
        }
        // Reduce to echelon form, mirroring in result
        for(size_t i=0; i<get_width(); i++) {
            result.multiply_row_scalar(i, 1/a(i,i));
            multiply_row_scalar(i, 1/a(i,i));
            for(size_t j=i+1; j<get_height(); j++) {
                result.add_row_multiple(i, j, -a(i,j));
                add_row_multiple(i, j, -a(i,j));
            }
        }
        // Back substitute, mirroring in result
        for(size_t i=get_width()-1; i != std::numeric_limits<size_t>::max(); i--) {
            for(int j=i-1; j != std::numeric_limits<size_t>::max(); j--) {
                result.add_row_multiple(i, j, -a(i,j));
                add_row_multiple(i, j, -a(i,j));
            }
        }
        // result is now the inverse
        return result;
    }

private:
    std::unique_ptr<T[]> data;
    size_t width, height;
};

template <typename T>
array2d<T> operator*(T scalar, const array2d<T>& a) {
    return a*scalar;
}


template <typename T>
ostream& operator<<(ostream& out, const array2d<T>& a) {
    out << "(";
    int i, j;
    for (j=0; j<a.get_height(); j++) {
        out << "(";
        for (i=0; i<a.get_width() - 1; i++) {
            out << a(i, j) << ", ";
        }
        if (j == a.get_height() - 1) {
            out << a(i, j) << "))" << endl;
        } else {
            out << a(i, j) << ")," << endl << " ";
        }
    }
    return out;
}

template <typename T>
class array3d
{
public:
    array3d(int width, int height, int depth)
    {
        this->width = width;
        this->height = height;
        this->depth = depth;
        data = new T[width * height * depth];
    }

    array3d(const array3d<T>& rhs)
    {
        width = rhs.width;
        height = rhs.height;
        depth = rhs.depth;
        data = new T[width * height * depth];
        for(int i=0; i<width; i++) {
            for(int j=0; j<height; j++) {
                for(int k=0; k<depth; k++) {
                    (*this)(i,j,k) = rhs.data[j*width*depth + i*depth + k];
                }
            }
        }
    }

    ~array3d()
    {
        delete [] data;
    }

    T& operator()(int col, int row, int layer)
    {
        return data[row*width*depth + col*depth + layer];
    }

    int get_width() { return width; }
    int get_height() { return height; }
    int get_depth() { return depth; }

private:
    T* data;
    int width, height, depth;
};

template <typename T>
ostream& operator<<(ostream& out, array3d<T>& a) {
    out << "(";
    int i, j, k;
    out << "(";
    for (j=0; j<=a.get_height() - 1; j++) {
        out << "(";
        for (i=0; i<=a.get_width() - 1; i++) {
            out << "(";
            for (k=0; k<=a.get_depth() - 1; k++) {
                out << a(i, j, k);
                if (k < a.get_depth() - 1) out << ", ";
            }
            out << ")";
            if (i < a.get_height() - 1) out << ",";
        }
        out << ")";
        if (j < a.get_height() - 1) out << ", " << endl;
    }
    out << ")" << endl;
    return out;
}

int compute_max_coarse_level(int width, int height) {
    // We want the coarsest layer to have at most MAX_PIXELS pixels
    const int MAX_PIXELS = 4000;
    int result = 0;
    while (width * height > MAX_PIXELS) {
        width  >>= 1;
        height >>= 1;
        result++;
    }
    return result;
}

void fill_random(array3d<double>& a) {
    for(int i=0; i<a.get_width(); i++) {
        for(int j=0; j<a.get_height(); j++) {
            for(int k=0; k<a.get_depth(); k++) {
                a(i,j,k) = ((double)rand())/RAND_MAX;
            }
        }
    }
}

double get_initial_temperature() {
    return 2.0; // TODO: Figure out what to make this
}

double get_final_temperature() {
    return 0.02; // TODO: Figure out what to make this
}

void random_permutation(int count, vector<int>& result) {
    result.clear();
    for(int i=0; i<count; i++) {
        result.push_back(i);
    }
    random_shuffle(result.begin(), result.end());
}

void random_permutation_2d(int width, int height, deque< pair<int, int> >& result) {
    vector<int> perm1d;
    random_permutation(width*height, perm1d);
    while(!perm1d.empty()) {
        int idx = perm1d.back();
        perm1d.pop_back();
        result.push_back(pair<int,int>(idx % width, idx / width));
    }
}


/// The b array is the element-wise product
void compute_b_array(array2d< vector_fixed<double, 3> >& filter_weights,
                     array2d< vector_fixed<double, 3> >& b)
{
    // Assume that the pixel i is always located at the center of b,
    // and vary pixel j's location through each location in b.
    int radius_width = (filter_weights.get_width() - 1)/2,
        radius_height = (filter_weights.get_height() - 1)/2;
    int offset_x = (b.get_width() - 1)/2 - radius_width;
    int offset_y = (b.get_height() - 1)/2 - radius_height;
    for(int j_y = 0; j_y < b.get_height(); j_y++) {
        for(int j_x = 0; j_x < b.get_width(); j_x++) {
            for(int k_y = 0; k_y < filter_weights.get_height(); k_y++) {
                for(int k_x = 0; k_x < filter_weights.get_width(); k_x++) {
                    if (k_x+offset_x >= j_x - radius_width &&
                        k_x+offset_x <= j_x + radius_width &&
                        k_y+offset_y >= j_y - radius_width &&
                        k_y+offset_y <= j_y + radius_width)
                    {
                        b(j_x,j_y) += filter_weights(k_x,k_y).direct_product(filter_weights(k_x+offset_x-j_x+radius_width,k_y+offset_y-j_y+radius_height));
                    }
                }
            }            
        }
    }
}

vector_fixed<double, 3> b_value(array2d< vector_fixed<double, 3> >& b,
                                  int i_x, int i_y, int j_x, int j_y)
{
    int radius_width = (b.get_width() - 1)/2,
        radius_height = (b.get_height() - 1)/2;
    int k_x = j_x - i_x + radius_width;
    int k_y = j_y - i_y + radius_height;
    if (k_x >= 0 && k_y >= 0 && k_x < b.get_width() && k_y < b.get_height())
        return b(k_x, k_y);
    else
        return vector_fixed<double, 3>();
}

void compute_a_image(array2d< vector_fixed<double, 3> >& image,
                     array2d< vector_fixed<double, 3> >& b,
                     array2d< vector_fixed<double, 3> >& a)
{
    int radius_width = (b.get_width() - 1)/2,
        radius_height = (b.get_height() - 1)/2;
    for(int i_y = 0; i_y < a.get_height(); i_y++) {
        for(int i_x = 0; i_x < a.get_width(); i_x++) {
            for(int j_y = i_y - radius_height; j_y <= i_y + radius_height; j_y++) {
                if (j_y < 0) j_y = 0;
                if (j_y >= a.get_height()) break;

                for(int j_x = i_x - radius_width; j_x <= i_x + radius_width; j_x++) {
                    if (j_x < 0) j_x = 0;
                    if (j_x >= a.get_width()) break;

                    a(i_x,i_y) += b_value(b, i_x, i_y, j_x, j_y).
                                      direct_product(image(j_x,j_y));
                }
            }
            a(i_x, i_y) *= -2.0;
        }
    }
}

void sum_coarsen(array2d< vector_fixed<double, 3> >& fine,
                 array2d< vector_fixed<double, 3> >& coarse)
{
    for(int y=0; y<coarse.get_height(); y++) {
        for(int x=0; x<coarse.get_width(); x++) {
            double divisor = 1.0;
            vector_fixed<double, 3> val = fine(x*2, y*2);
            if (x*2 + 1 < fine.get_width())  {
                divisor += 1; val += fine(x*2 + 1, y*2);
            }
            if (y*2 + 1 < fine.get_height()) {
                divisor += 1; val += fine(x*2, y*2 + 1);
            }
            if (x*2 + 1 < fine.get_width() &&
                y*2 + 1 < fine.get_height()) {
                divisor += 1; val += fine(x*2 + 1, y*2 + 1);
            }
            coarse(x, y) = /*(1/divisor)**/val;
        }
    }
}

template <typename T, size_t length>
array2d<T> extract_vector_layer_2d(array2d< vector_fixed<T, length> > s, int k)
{
    array2d<T> result(s.get_width(), s.get_height());
    for(int i=0; i < s.get_width(); i++) {
        for(int j=0; j < s.get_height(); j++) {
            result(i,j) = s(i,j)(k);
        }
    }
    return result;
}

template <typename T, size_t length>
vector<T> extract_vector_layer_1d(vector< vector_fixed<T, length> > s, int k)
{
    vector<T> result;
    for(unsigned int i=0; i < s.size(); i++) {
        result.push_back(s[i](k));
    }
    return result;
}

int best_match_color(array3d<double>& vars, int i_x, int i_y,
                     vector< vector_fixed<double, 3> >& palette)
{
    int max_v = 0;
    double max_weight = vars(i_x, i_y, 0);
    for (unsigned int v=1; v < palette.size(); v++) {
        if (vars(i_x, i_y, v) > max_weight) {
            max_v = v;
            max_weight = vars(i_x, i_y, v);
        }
    }
    return max_v;
}

void zoom_double(array3d<double>& small, array3d<double>& big)
{
    // Simple scaling of the weights array based on mixing the four
    // pixels falling under each fine pixel, weighted by area.
    // To mix the pixels a little, we assume each fine pixel
    // is 1.2 fine pixels wide and high.
    for(int y=0; y<big.get_height()/2*2; y++) {
        for(int x=0; x<big.get_width()/2*2; x++) {
            double left = max(0.0, (x-0.1)/2.0), right  = min(small.get_width()-0.001, (x+1.1)/2.0);
            double top  = max(0.0, (y-0.1)/2.0), bottom = min(small.get_height()-0.001, (y+1.1)/2.0);
            int x_left = (int)floor(left), x_right  = (int)floor(right);
            int y_top  = (int)floor(top),  y_bottom = (int)floor(bottom);
            double area = (right-left)*(bottom-top);
            double top_left_weight  = (ceil(left) - left)*(ceil(top) - top)/area;
            double top_right_weight = (right - floor(right))*(ceil(top) - top)/area;
            double bottom_left_weight  = (ceil(left) - left)*(bottom - floor(bottom))/area;
            double bottom_right_weight = (right - floor(right))*(bottom - floor(bottom))/area;
            double top_weight     = (right-left)*(ceil(top) - top)/area;
            double bottom_weight  = (right-left)*(bottom - floor(bottom))/area;
            double left_weight    = (bottom-top)*(ceil(left) - left)/area;
            double right_weight   = (bottom-top)*(right - floor(right))/area;
            for(int z=0; z<big.get_depth(); z++) {
                if (x_left == x_right && y_top == y_bottom) {
                    big(x, y, z) = small(x_left, y_top, z);
                } else if (x_left == x_right) {
                    big(x, y, z) = top_weight*small(x_left, y_top, z) +
                                   bottom_weight*small(x_left, y_bottom, z);
                } else if (y_top == y_bottom) {
                    big(x, y, z) = left_weight*small(x_left, y_top, z) +
                                   right_weight*small(x_right, y_top, z);
                } else {
                    big(x, y, z) = top_left_weight*small(x_left, y_top, z) +
                                   top_right_weight*small(x_right, y_top, z) +
                                   bottom_left_weight*small(x_left, y_bottom, z) +
                                   bottom_right_weight*small(x_right, y_bottom, z);
                }
            }
        }
    }
}

void compute_initial_s(array2d< vector_fixed<double,3> >& s,
                       array3d<double>& coarse_variables,
                       array2d< vector_fixed<double, 3> >& b)
{
    size_t palette_size  = s.get_width();
    size_t coarse_width  = coarse_variables.get_width();
    size_t coarse_height = coarse_variables.get_height();
    size_t center_x = (b.get_width()-1)/2, center_y = (b.get_height()-1)/2;
    vector_fixed<double,3> center_b = b_value(b,0,0,0,0);
    vector_fixed<double,3> zero_vector;
    for (size_t v=0; v<palette_size; v++) {
        for (size_t alpha=v; alpha<palette_size; alpha++) {
            s(v,alpha) = zero_vector;
        }
    }
    for (size_t i_y=0; i_y<coarse_height; i_y++) {
        for (size_t i_x=0; i_x<coarse_width; i_x++) {
            size_t max_j_x = min(coarse_width,  i_x - center_x + b.get_width());
            size_t max_j_y = min(coarse_height, i_y - center_y + b.get_height());
            for (size_t j_y=(i_y >= center_y ? i_y - center_y : 0); j_y<max_j_y; j_y++) {
                for (size_t j_x=(i_x >= center_x ? i_x - center_x : 0); j_x<max_j_x; j_x++) {
                    if (i_x == j_x && i_y == j_y) continue;
                    vector_fixed<double,3> b_ij = b_value(b,i_x,i_y,j_x,j_y);
                    for (size_t v=0; v<palette_size; v++) {
                        for (size_t alpha=v; alpha<palette_size; alpha++) {
                            double mult = coarse_variables(i_x,i_y,v)*
                                          coarse_variables(j_x,j_y,alpha);
                            s(v,alpha)(0) += mult * b_ij(0);
                            s(v,alpha)(1) += mult * b_ij(1);
                            s(v,alpha)(2) += mult * b_ij(2);
                        }
                    }
                }
            }            
            for (size_t v=0; v<palette_size; v++) {
                s(v,v) += coarse_variables(i_x,i_y,v)*center_b;
            }
        }
    }
}

void update_s(array2d< vector_fixed<double,3> >& s,
              array3d<double>& coarse_variables,
              array2d< vector_fixed<double, 3> >& b,
              int j_x, int j_y, int alpha,
              double delta)
{
    int palette_size  = s.get_width();
    int coarse_width  = coarse_variables.get_width();
    int coarse_height = coarse_variables.get_height();
    int center_x = (b.get_width()-1)/2, center_y = (b.get_height()-1)/2;
    int max_i_x = min(coarse_width,  j_x + center_x + 1);
    int max_i_y = min(coarse_height, j_y + center_y + 1);
    for (int i_y=max(0, j_y - center_y); i_y<max_i_y; i_y++) {
        for (int i_x=max(0, j_x - center_x); i_x<max_i_x; i_x++) {
            vector_fixed<double,3> delta_b_ij = delta*b_value(b,i_x,i_y,j_x,j_y);
            if (i_x == j_x && i_y == j_y) continue;
            for (int v=0; v <= alpha; v++) {
                double mult = coarse_variables(i_x,i_y,v);
                s(v,alpha)(0) += mult * delta_b_ij(0);
                s(v,alpha)(1) += mult * delta_b_ij(1);
                s(v,alpha)(2) += mult * delta_b_ij(2);
            }
            for (int v=alpha; v<palette_size; v++) {
                double mult = coarse_variables(i_x,i_y,v);
                s(alpha,v)(0) += mult * delta_b_ij(0);
                s(alpha,v)(1) += mult * delta_b_ij(1);
                s(alpha,v)(2) += mult * delta_b_ij(2);
            }
        }
    }
    s(alpha,alpha) += delta*b_value(b,0,0,0,0);
}

void refine_palette(array2d< vector_fixed<double,3> >& s,
                    array3d<double>& coarse_variables,
                    array2d< vector_fixed<double, 3> >& a,
                    vector< vector_fixed<double, 3> >& palette)
{
    // We only computed the half of S above the diagonal - reflect it
    for (int v=0; v<s.get_width(); v++) {
        for (int alpha=0; alpha<v; alpha++) {
            s(v,alpha) = s(alpha,v);
        }
    }

    vector< vector_fixed<double,3> > r(palette.size());
    for (unsigned int v=0; v<palette.size(); v++) {
        for (int i_y=0; i_y<coarse_variables.get_height(); i_y++) {
            for (int i_x=0; i_x<coarse_variables.get_width(); i_x++) {
                r[v] += coarse_variables(i_x,i_y,v)*a(i_x,i_y);
            }
        }
    }

    for (unsigned int k=0; k<3; k++) {
        array2d<double> S_k = extract_vector_layer_2d(s, k);
        vector<double> R_k = extract_vector_layer_1d(r, k);
        vector<double> palette_channel = -1.0*((2.0*S_k).matrix_inverse())*R_k;
        for (unsigned int v=0; v<palette.size(); v++) {
            double val = palette_channel[v];
            if (val < 0) val = 0;
            if (val > 1) val = 1;
            palette[v](k) = val;
        }                
    }

#if TRACE
    for (unsigned int v=0; v<palette.size(); v++) {
        cout << palette[v] << endl;
    }
#endif
}

void compute_initial_j_palette_sum(array2d< vector_fixed<double, 3> >& j_palette_sum,
                                   array3d<double>& coarse_variables,
                                   vector< vector_fixed<double, 3> >& palette)
{
     for (int j_y=0; j_y<coarse_variables.get_height(); j_y++) {
         for (int j_x=0; j_x<coarse_variables.get_width(); j_x++) {
             vector_fixed<double, 3> palette_sum = vector_fixed<double, 3>();
             for (unsigned int alpha=0; alpha < palette.size(); alpha++) {
                 palette_sum += coarse_variables(j_x,j_y,alpha)*palette[alpha];
             }
             j_palette_sum(j_x, j_y) = palette_sum;
         }
     }
}

void spatial_color_quant(array2d< vector_fixed<double, 3> >& image,
                         array2d< vector_fixed<double, 3> >& filter_weights,
                         array2d< int >& quantized_image,
                         vector< vector_fixed<double, 3> >& palette,
                         array3d<double>*& p_coarse_variables,
                         double initial_temperature,
                         double final_temperature,
                         int temps_per_level,
                         int repeats_per_temp)
{
    int max_coarse_level = //1;
        compute_max_coarse_level(image.get_width(), image.get_height());
    p_coarse_variables = new array3d<double>(
        image.get_width()  >> max_coarse_level,
        image.get_height() >> max_coarse_level,
        palette.size());
    // For syntactic convenience
    array3d<double>& coarse_variables = *p_coarse_variables;
    fill_random(coarse_variables);

    double temperature = initial_temperature;

    // Compute a_i, b_{ij} according to (11)
    int extended_neighborhood_width = filter_weights.get_width()*2 - 1;
    int extended_neighborhood_height = filter_weights.get_height()*2 - 1;
    array2d< vector_fixed<double, 3> > b0(extended_neighborhood_width,
                                          extended_neighborhood_height);
    compute_b_array(filter_weights, b0);

    array2d< vector_fixed<double, 3> > a0(image.get_width(), image.get_height());
    compute_a_image(image, b0, a0);

    // Compute a_I^l, b_{IJ}^l according to (18)
    vector< array2d< vector_fixed<double, 3> > > a_vec, b_vec;
    a_vec.push_back(a0);
    b_vec.push_back(b0);

    int coarse_level;
    for(coarse_level=1; coarse_level <= max_coarse_level; coarse_level++)
    {
        size_t radius_width  = (filter_weights.get_width() - 1)/2,
            radius_height = (filter_weights.get_height() - 1)/2;
        array2d< vector_fixed<double, 3> >
            bi(max(static_cast<size_t>(3), b_vec.back().get_width()-2),
               max(static_cast<size_t>(3), b_vec.back().get_height()-2));
        for(size_t J_y=0; J_y<bi.get_height(); J_y++) {
            for(size_t J_x=0; J_x<bi.get_width(); J_x++) {
                for(size_t i_y=radius_height*2; i_y<radius_height*2+2; i_y++) {
                    for(size_t i_x=radius_width*2; i_x<radius_width*2+2; i_x++) {
                        for(size_t j_y=J_y*2; j_y<J_y*2+2; j_y++) {
                            for(size_t j_x=J_x*2; j_x<J_x*2+2; j_x++) {
                                bi(J_x,J_y) += b_value(b_vec.back(), i_x, i_y, j_x, j_y);
                            }
                        }
                    }
                }
            }
        }
        b_vec.push_back(bi);

        array2d< vector_fixed<double, 3> >
            ai(image.get_width() >> coarse_level,
               image.get_height() >> coarse_level);
        sum_coarsen(a_vec.back(), ai);
        a_vec.push_back(ai);
    }

    // Multiscale annealing
    coarse_level = max_coarse_level;
    const int iters_per_level = temps_per_level;
    double temperature_multiplier = pow(final_temperature/initial_temperature, 1.0/(max(3, max_coarse_level*iters_per_level)));
#if TRACE
    cout << "Temperature multiplier: " << temperature_multiplier << endl;
#endif
    int iters_at_current_level = 0;
    bool skip_palette_maintenance = false;
    array2d< vector_fixed<double,3> > s(palette.size(), palette.size());
    compute_initial_s(s, coarse_variables, b_vec[coarse_level]);
    array2d< vector_fixed<double, 3> >* j_palette_sum =
        new array2d< vector_fixed<double, 3> >(coarse_variables.get_width(), coarse_variables.get_height());
    compute_initial_j_palette_sum(*j_palette_sum, coarse_variables, palette);
    while (coarse_level >= 0 || temperature > final_temperature) {
        // Need to reseat this reference in case we changed p_coarse_variables
        array3d<double>& coarse_variables = *p_coarse_variables;
        array2d< vector_fixed<double, 3> >& a = a_vec[coarse_level];
        array2d< vector_fixed<double, 3> >& b = b_vec[coarse_level];
        vector_fixed<double,3> middle_b = b_value(b,0,0,0,0);
#if TRACE
        cout << "Temperature: " << temperature << endl;
#endif
        size_t center_x = (b.get_width()-1)/2, center_y = (b.get_height()-1)/2;
        int step_counter = 0;
        for(int repeat=0; repeat<repeats_per_temp; repeat++)
        {
            int pixels_changed = 0, pixels_visited = 0;
            deque< pair<int, int> > visit_queue;
            random_permutation_2d(coarse_variables.get_width(), coarse_variables.get_height(), visit_queue);

            // Compute 2*sum(j in extended neighborhood of i, j != i) b_ij

            while(!visit_queue.empty())
            {
                // If we get to 10% above initial size, just revisit them all
                if ((int)visit_queue.size() > coarse_variables.get_width()*coarse_variables.get_height()*11/10) {
                    visit_queue.clear();
                    random_permutation_2d(coarse_variables.get_width(), coarse_variables.get_height(), visit_queue);
                }

                int i_x = visit_queue.front().first, i_y = visit_queue.front().second;
                visit_queue.pop_front();

                // Compute (25)
                vector_fixed<double,3> p_i;
                for (int y=0; y<b.get_height(); y++) {
                    for (int x=0; x<b.get_width(); x++) {
                        int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
                        if (i_x == j_x && i_y == j_y) continue;
                        if (j_x < 0 || j_y < 0 || j_x >= coarse_variables.get_width() || j_y >= coarse_variables.get_height()) continue;
                        vector_fixed<double,3> b_ij = b_value(b, i_x, i_y, j_x, j_y);
                        vector_fixed<double,3> j_pal = (*j_palette_sum)(j_x,j_y);
                        p_i(0) += b_ij(0)*j_pal(0);
                        p_i(1) += b_ij(1)*j_pal(1);
                        p_i(2) += b_ij(2)*j_pal(2);
                    }
                }
                p_i *= 2.0;
                p_i += a(i_x, i_y);

                vector<double> meanfield_logs, meanfields;
                double max_meanfield_log = -numeric_limits<double>::infinity();
                double meanfield_sum = 0.0;
                for (unsigned int v=0; v < palette.size(); v++) {
                    // Update m_{pi(i)v}^I according to (23)
                    // We can subtract an arbitrary factor to prevent overflow,
                    // since only the weight relative to the sum matters, so we
                    // will choose a value that makes the maximum e^100.
                    meanfield_logs.push_back(-(palette[v].dot_product(
                        p_i + middle_b.direct_product(
                            palette[v])))/temperature);
                    if (meanfield_logs.back() > max_meanfield_log) {
                        max_meanfield_log = meanfield_logs.back();
                    }
                }
                for (unsigned int v=0; v < palette.size(); v++) {
                    meanfields.push_back(exp(meanfield_logs[v]-max_meanfield_log+100));
                    meanfield_sum += meanfields.back();
                }
                if (meanfield_sum == 0) {
                    cout << "Fatal error: Meanfield sum underflowed. Please contact developer." << endl;
                    exit(-1);
                }
                int old_max_v = best_match_color(coarse_variables, i_x, i_y, palette);
                vector_fixed<double,3>& j_pal = (*j_palette_sum)(i_x,i_y);
                for (unsigned int v=0; v < palette.size(); v++) {
                    double new_val = meanfields[v]/meanfield_sum;
                    // Prevent the matrix S from becoming singular
                    if (new_val <= 0) new_val = 1e-10;
                    if (new_val >= 1) new_val = 1 - 1e-10;
                    double delta_m_iv = new_val - coarse_variables(i_x,i_y,v);
                    coarse_variables(i_x,i_y,v) = new_val;
                    j_pal(0) += delta_m_iv*palette[v](0);
                    j_pal(1) += delta_m_iv*palette[v](1);
                    j_pal(2) += delta_m_iv*palette[v](2);
                    if (abs(delta_m_iv) > 0.001 && !skip_palette_maintenance) {
                        update_s(s, coarse_variables, b, i_x, i_y, v, delta_m_iv);
                    }
                }
                int max_v = best_match_color(coarse_variables, i_x, i_y, palette);
                // Only consider it a change if the colors are different enough
                if ((palette[max_v]-palette[old_max_v]).norm_squared() >= 1.0/(255.0*255.0)) {
                    pixels_changed++;
                    // We don't add the outer layer of pixels , because
                    // there isn't much weight there, and if it does need
                    // to be visited, it'll probably be added when we visit
                    // neighboring pixels.
                    // The commented out loops are faster but cause a little bit of distortion
                    //for (int y=center_y-1; y<center_y+1; y++) {
                    //   for (int x=center_x-1; x<center_x+1; x++) {
                    for (int y=min(static_cast<size_t>(1),center_y-1); y<max(b.get_height()-1,center_y+1); y++) {
                        for (int x=min(static_cast<size_t>(1),center_x-1); x<max(b.get_width()-1,center_x+1); x++) {
                            int j_x = x - center_x + i_x, j_y = y - center_y + i_y;
                            if (j_x < 0 || j_y < 0 || j_x >= coarse_variables.get_width() || j_y >= coarse_variables.get_height()) continue;
                            visit_queue.push_back(pair<int,int>(j_x,j_y));
                        }
                    }
                }
                pixels_visited++;

                // Show progress with dots - in a graphical interface,
                // we'd show progressive refinements of the image instead,
                // and maybe a palette preview.
                step_counter++;
                if ((step_counter % 10000) == 0) {
                    cout << ".";
                    cout.flush();
#if TRACE
                    cout << visit_queue.size();
#endif
                }
            }
#if TRACE
            cout << "Pixels changed: " << pixels_changed << endl;
#endif
            if (skip_palette_maintenance) {
                compute_initial_s(s, *p_coarse_variables, b_vec[coarse_level]);
            }
            refine_palette(s, coarse_variables, a, palette);
            compute_initial_j_palette_sum(*j_palette_sum, coarse_variables, palette);
        }

        iters_at_current_level++;
        skip_palette_maintenance = false;
        if ((temperature <= final_temperature || coarse_level > 0) &&
            iters_at_current_level >= iters_per_level)
        {
            coarse_level--;
            if (coarse_level < 0) break;
            array3d<double>* p_new_coarse_variables = new array3d<double>(
                image.get_width()  >> coarse_level,
                image.get_height() >> coarse_level,
                palette.size());
            zoom_double(coarse_variables, *p_new_coarse_variables);
            delete p_coarse_variables;
            p_coarse_variables = p_new_coarse_variables;
            iters_at_current_level = 0;
            delete j_palette_sum;
            j_palette_sum = new array2d< vector_fixed<double, 3> >((*p_coarse_variables).get_width(), (*p_coarse_variables).get_height());
            compute_initial_j_palette_sum(*j_palette_sum, *p_coarse_variables, palette);
            skip_palette_maintenance = true;
#ifdef TRACE
            cout << "Image size: " << p_coarse_variables->get_width() << " " << p_coarse_variables->get_height() << endl;
#endif
        }
        if (temperature > final_temperature) {
            temperature *= temperature_multiplier;
        }
    }

    // This is normally not used, but is handy sometimes for debugging
    while (coarse_level > 0) {
        coarse_level--;
        array3d<double>* p_new_coarse_variables = new array3d<double>(
            image.get_width()  >> coarse_level,
            image.get_height() >> coarse_level,
            palette.size());
        zoom_double(*p_coarse_variables, *p_new_coarse_variables);
        delete p_coarse_variables;
        p_coarse_variables = p_new_coarse_variables;
    }

    {
    // Need to reseat this reference in case we changed p_coarse_variables
    array3d<double>& coarse_variables = *p_coarse_variables;

    for(size_t i_x = 0; i_x < image.get_width(); i_x++) {
        for(size_t i_y = 0; i_y < image.get_height(); i_y++) {
            quantized_image(i_x,i_y) =
                best_match_color(coarse_variables, i_x, i_y, palette);
        }
    }
    for (unsigned int v=0; v<palette.size(); v++) {
        for (unsigned int k=0; k<3; k++) {
            if (palette[v](k) > 1.0) palette[v](k) = 1.0;
            if (palette[v](k) < 0.0) palette[v](k) = 0.0;
        }
#ifdef TRACE
        cout << palette[v] << endl;
#endif
    }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 1 + 5 || argc > 1 + 7) {
        printf("Usage: spatial_color_quant <source image.rgb> <width> <height> <desired palette size> <output image.rgb> [dithering level] [filter size (1/3/5)]\n");
        return -1;
    }

    srand(time(NULL));

    const int width = atoi(argv[2]), height = atoi(argv[3]);
    if (width <= 0 || height <= 0) {
        printf("Must specify a valid positive image width and height.\n");
        return -1;
    }

    array2d< vector_fixed<double, 3> > image(width, height);
    array2d< vector_fixed<double, 3> > filter1_weights(1, 1);
    array2d< vector_fixed<double, 3> > filter3_weights(3, 3);
    array2d< vector_fixed<double, 3> > filter5_weights(5, 5);
    array2d< int > quantized_image(width, height);
    vector< vector_fixed<double, 3> > palette;

    for(int k=0; k<3; k++) {
        filter1_weights(0,0)(k) = 1.0;
    }

    int num_colors = atoi(argv[4]);
    if (num_colors <= 1 || num_colors > 256) {
        printf("Number of colors must be at least 2 and no more than 256.\n");
        return -1;
    }
    for (int i=0; i<atoi(argv[4]); i++) {
        vector_fixed<double, 3> v;
        v(0) = ((double)rand())/RAND_MAX;
        v(1) = ((double)rand())/RAND_MAX;
        v(2) = ((double)rand())/RAND_MAX;
        palette.push_back(v);
    }

#if TRACE
    for (unsigned int v=0; v<palette.size(); v++) {
        cout << palette[v] << endl;
    }
#endif

    {
    unsigned char c[3];
    FILE* in = fopen(argv[1], "rb");
    if (in == NULL) {
        printf("Could not open input file '%s'.\n", argv[1]);
        return -1;
    }
    for(int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            fread(c, 3, 1, in);
            for(int ci=0; ci<3; ci++) {
                image(x,y)(ci) = c[ci]/((double)255);
            }
        }
    }
    fclose(in);
    }

    // Check the output file before we begin the long part
    FILE* out = fopen(argv[5], "wb");
    if (out == NULL) {
        printf("Could not open output file '%s'.\n", argv[5]);
        return -1;
    }
    fclose(out);
    

    array3d<double>* coarse_variables;
    double dithering_level = 0.09*log((double)image.get_width()*image.get_height()) - 0.04*log((double)palette.size()) + 0.001;
    if (argc > 6) {
        dithering_level = atof(argv[6]); 
        if (dithering_level <= 0.0) {
            printf("Dithering level must be more than zero.\n");
            return -1;
        }
    }
    int filter_size = 3;
    if (argc > 7) {
        filter_size = atoi(argv[7]); 
        if (filter_size != 1 && filter_size != 3 && filter_size != 5) {
            printf("Filter size must be one of 1, 3, or 5.\n");
            return -1;
        }
    }

    double stddev = dithering_level;
    double sum = 0.0;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            for(int k=0; k<3; k++) {
                sum += filter3_weights(i,j)(k) =
                    exp(-sqrt((double)((i-1)*(i-1) + (j-1)*(j-1)))/(stddev*stddev));
            }
        }
    }
    sum /= 3;
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            for(int k=0; k<3; k++) {
                filter3_weights(i,j)(k) /= sum;
            }
        }
    }
    sum = 0.0;
    for(int i=0; i<5; i++) {
        for(int j=0; j<5; j++) {
            for(int k=0; k<3; k++) {
                sum += filter5_weights(i,j)(k) =
                    exp(-sqrt((double)((i-2)*(i-2) + (j-2)*(j-2)))/(stddev*stddev));
            }
        }
    }
    sum /= 3;
    for(int i=0; i<5; i++) {
        for(int j=0; j<5; j++) {
            for(int k=0; k<3; k++) {
                filter5_weights(i,j)(k) /= sum;
            }
        }
    }

    array2d< vector_fixed<double, 3> >* filters[] =
        {NULL, &filter1_weights, NULL, &filter3_weights,
         NULL, &filter5_weights};
    spatial_color_quant(image, *filters[filter_size], quantized_image, palette, coarse_variables, 1.0, 0.001, 3, 1);
    //spatial_color_quant(image, filter3_weights, quantized_image, palette, coarse_variables, 0.05, 0.02);

    cout << endl;

    {
    FILE* out = fopen(argv[5], "wb");
    if (out == NULL) {
        printf("Could not open output file '%s'.\n", argv[5]);
        return -1;
    }
    unsigned char c[3] = {0,0,0};
    for(int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            c[0] = (unsigned char)(255*palette[quantized_image(x,y)](0));
            c[1] = (unsigned char)(255*palette[quantized_image(x,y)](1));
            c[2] = (unsigned char)(255*palette[quantized_image(x,y)](2));
            fwrite(c, 3, 1, out);
        }
    }
    fclose(out);
    }

    return 0;
}
