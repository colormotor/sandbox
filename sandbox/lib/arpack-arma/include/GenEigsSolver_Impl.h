// Copyright (C) 2015 Yixuan Qiu
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Arnoldi factorization starting from step-k
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::factorize_from(int from_k, int to_m, const Vector &fk)
{
    if(to_m <= from_k) return;

    fac_f = fk;

    Vector w(dim_n);
    Scalar beta = std::sqrt(arma::dot(fac_f, fac_f));
    // Keep the upperleft k x k submatrix of H and set other elements to 0
    fac_H.tail_cols(ncv - from_k).zeros();
    fac_H.submat(arma::span(from_k, ncv - 1), arma::span(0, from_k - 1)).zeros();
    for(int i = from_k; i <= to_m - 1; i++)
    {
        bool restart = false;
        // If beta = 0, then the next V is not full rank
        // We need to generate a new residual vector that is orthogonal
        // to the current V, which we call a restart
        if(beta < prec)
        {
            fac_f.randu();
            // f <- f - V * V' * f, so that f is orthogonal to V
            Matrix Vs(fac_V.memptr(), dim_n, i, false); // First i columns
            Vector Vf = Vs.t() * fac_f;
            fac_f -= Vs * Vf;
            // beta <- ||f||
            beta = std::sqrt(arma::dot(fac_f, fac_f));

            restart = true;
        }

        // v <- f / ||f||
        fac_V.col(i) = fac_f / beta; // The (i+1)-th column

        // Note that H[i+1, i] equals to the unrestarted beta
        if(restart)
            fac_H(i, i - 1) = 0.0;
        else
            fac_H(i, i - 1) = beta;

        // w <- A * v, v = fac_V.col(i)
        op->perform_op(fac_V.colptr(i), w.memptr());
        nmatop++;

        // First i+1 columns of V
        Matrix Vs(fac_V.memptr(), dim_n, i + 1, false);
        // h = fac_H(0:i, i)
        Vector h(fac_H.colptr(i), i + 1, false);
        // h <- V' * w
        h = Vs.t() * w;

        // f <- w - V * h
        fac_f = w - Vs * h;
        beta = std::sqrt(arma::dot(fac_f, fac_f));

        if(beta > 0.717 * std::sqrt(arma::dot(h, h)))
            continue;

        // f/||f|| is going to be the next column of V, so we need to test
        // whether V' * (f/||f||) ~= 0
        Vector Vf = Vs.t() * fac_f;
        // If not, iteratively correct the residual
        int count = 0;
        while(count < 5 && arma::abs(Vf).max() > prec * beta)
        {
            // f <- f - V * Vf
            fac_f -= Vs * Vf;
            // h <- h + Vf
            h += Vf;
            // beta <- ||f||
            beta = std::sqrt(arma::dot(fac_f, fac_f));

            Vf = Vs.t() * fac_f;
            count++;
        }
    }
}

// Implicitly restarted Arnoldi factorization
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::restart(int k)
{
    if(k >= ncv)
        return;

    DoubleShiftQR<Scalar> decomp_ds(ncv);
    UpperHessenbergQR<Scalar> decomp;
    Matrix Q(ncv, ncv, arma::fill::eye);

    for(int i = k; i < ncv; i++)
    {
        if(is_complex(ritz_val[i], prec) && is_conj(ritz_val[i], ritz_val[i + 1], prec))
        {
            // H - mu * I = Q1 * R1
            // H <- R1 * Q1 + mu * I = Q1' * H * Q1
            // H - conj(mu) * I = Q2 * R2
            // H <- R2 * Q2 + conj(mu) * I = Q2' * H * Q2
            //
            // (H - mu * I) * (H - conj(mu) * I) = Q1 * Q2 * R2 * R1 = Q * R
            Scalar s = 2 * ritz_val[i].real();
            Scalar t = std::norm(ritz_val[i]);

            decomp_ds.compute(fac_H, s, t);

            // Q -> Q * Qi
            decomp_ds.apply_YQ(Q);
            //decomp_ds.apply_YQ(fac_V, ncv);
            // H -> Q'HQ
            // Matrix Q(ncv, ncv, arma::fill::eye);
            // decomp_ds.apply_YQ(Q);
            // fac_H = Q.t() * fac_H * Q;
            fac_H = decomp_ds.matrix_QtHQ();

            i++;
        } else {
            // QR decomposition of H - mu * I, mu is real
            fac_H.diag() -= ritz_val[i].real();
            decomp.compute(fac_H);

            // Q -> Q * Qi
            decomp.apply_YQ(Q);
            // H -> Q'HQ = RQ + mu * I
            fac_H = decomp.matrix_RQ();
            fac_H.diag() += ritz_val[i].real();
        }
    }
    // V -> VQ
    // Q has some elements being zero
    // The first (ncv - k + i) elements of the i-th column of Q are non-zero
    Matrix Vs(dim_n, k + 1);
    int nnz;
    for(int i = 0; i < k; i++)
    {
        nnz = ncv - k + i + 1;
        Matrix V(fac_V.memptr(), dim_n, nnz, false);
        Vector q(Q.colptr(i), nnz, false);
        Vector v(Vs.colptr(i), dim_n, false);
        v = V * q;
    }
    Vs.col(k) = fac_V * Q.col(k);
    fac_V.head_cols(k + 1) = Vs;

    Vector fk = fac_f * Q(ncv - 1, k - 1) + fac_V.col(k) * fac_H(k, k - 1);
    factorize_from(k, ncv, fk);
    retrieve_ritzpair();
}

// Calculate the number of converged Ritz values
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int GenEigsSolver<Scalar, SelectionRule, OpType>::num_converged(Scalar tol)
{
    // thresh = tol * max(prec, abs(theta)), theta for ritz value
    const Scalar f_norm = arma::norm(fac_f);
    for(int i = 0; i < nev; i++)
    {
        Scalar thresh = tol * std::max(prec, std::abs(ritz_val[i]));
        Scalar resid = std::abs(ritz_vec(ncv - 1, i)) * f_norm;
        ritz_conv[i] = (resid < thresh);
    }

    return std::count(ritz_conv.begin(), ritz_conv.end(), true);
}

// Return the adjusted nev for restarting
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int GenEigsSolver<Scalar, SelectionRule, OpType>::nev_adjusted(int nconv)
{
    int nev_new = nev;

    // Increase nev by one if ritz_val[nev - 1] and
    // ritz_val[nev] are conjugate pairs
    if(is_complex(ritz_val[nev - 1], prec) &&
       is_conj(ritz_val[nev - 1], ritz_val[nev], prec))
    {
        nev_new = nev + 1;
    }
    // Adjust nev_new again, according to dnaup2.f line 660~674 in ARPACK
    nev_new = nev_new + std::min(nconv, (ncv - nev_new) / 2);
    if(nev_new == 1 && ncv >= 6)
        nev_new = ncv / 2;
    else if(nev_new == 1 && ncv > 3)
        nev_new = 2;

    if(nev_new > ncv - 2)
        nev_new = ncv - 2;

    // Examine conjugate pairs again
    if(is_complex(ritz_val[nev_new - 1], prec) &&
       is_conj(ritz_val[nev_new - 1], ritz_val[nev_new], prec))
    {
        nev_new++;
    }

    return nev_new;
}

// Retrieve and sort ritz values and ritz vectors
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::retrieve_ritzpair()
{
    /*ComplexVector evals(ncv);
    ComplexMatrix evecs(ncv, ncv);
    arma::eig_gen(evals, evecs, fac_H);*/
    UpperHessenbergEigen<Scalar> decomp(fac_H);
    ComplexVector evals = decomp.eigenvalues();
    ComplexMatrix evecs = decomp.eigenvectors();

    SortEigenvalue<Complex, SelectionRule> sorting(evals.memptr(), evals.n_elem);
    std::vector<int> ind = sorting.index();

    // Copy the ritz values and vectors to ritz_val and ritz_vec, respectively
    for(int i = 0; i < ncv; i++)
    {
        ritz_val[i] = evals[ind[i]];
    }
    for(int i = 0; i < nev; i++)
    {
        ritz_vec.col(i) = evecs.col(ind[i]);
    }
}



// Sort the first nev Ritz pairs in decreasing magnitude order
// This is used to return the final results
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::sort_ritzpair()
{
    SortEigenvalue<Complex, LARGEST_MAGN> sorting(ritz_val.memptr(), nev);
    std::vector<int> ind = sorting.index();

    ComplexVector new_ritz_val(ncv);
    ComplexMatrix new_ritz_vec(ncv, nev);
    BoolVector new_ritz_conv(nev);

    for(int i = 0; i < nev; i++)
    {
        new_ritz_val[i] = ritz_val[ind[i]];
        new_ritz_vec.col(i) = ritz_vec.col(ind[i]);
        new_ritz_conv[i] = ritz_conv[ind[i]];
    }

    ritz_val.swap(new_ritz_val);
    ritz_vec.swap(new_ritz_vec);
    ritz_conv.swap(new_ritz_conv);
}



// Initialization and clean-up
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::init(Scalar *init_resid)
{
    // Reset all matrices/vectors to zero
    fac_V.zeros(dim_n, ncv);
    fac_H.zeros(ncv, ncv);
    fac_f.zeros(dim_n);
    ritz_val.zeros(ncv);
    ritz_vec.zeros(ncv, nev);
    ritz_conv.assign(nev, false);

    nmatop = 0;
    niter = 0;

    Vector r(init_resid, dim_n, false);
    // The first column of fac_V
    Vector v(fac_V.colptr(0), dim_n, false);
    Scalar rnorm = arma::norm(r);
    if(rnorm < prec)
        throw std::invalid_argument("initial residual vector cannot be zero");
    v = r / rnorm;

    Vector w(dim_n);
    op->perform_op(v.memptr(), w.memptr());
    nmatop++;

    fac_H(0, 0) = arma::dot(v, w);
    fac_f = w - v * fac_H(0, 0);
}

// Initialization with random initial coefficients
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline void GenEigsSolver<Scalar, SelectionRule, OpType>::init()
{
    Vector init_resid(dim_n, arma::fill::randu);
    init_resid -= 0.5;
    init(init_resid.memptr());
}

// Compute Ritz pairs and return the number of converged eigenvalues
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline int GenEigsSolver<Scalar, SelectionRule, OpType>::compute(int maxit, Scalar tol)
{
    // The m-step Arnoldi factorization
    factorize_from(1, ncv, fac_f);
    retrieve_ritzpair();
    // Restarting
    int i, nconv = 0, nev_adj;
    for(i = 0; i < maxit; i++)
    {
        nconv = num_converged(tol);
        if(nconv >= nev)
            break;

        nev_adj = nev_adjusted(nconv);
        restart(nev_adj);
    }
    // Sorting results
    sort_ritzpair();

    niter += i + 1;

    return std::min(nev, nconv);
}

// Return converged eigenvalues
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename GenEigsSolver<Scalar, SelectionRule, OpType>::ComplexVector GenEigsSolver<Scalar, SelectionRule, OpType>::eigenvalues()
{
    int nconv = std::count(ritz_conv.begin(), ritz_conv.end(), true);
    ComplexVector res(nconv);

    if(!nconv)
        return res;

    int j = 0;
    for(int i = 0; i < nev; i++)
    {
        if(ritz_conv[i])
        {
            res[j] = ritz_val[i];
            j++;
        }
    }

    return res;
}

// Return converged eigenvectors
template < typename Scalar,
           int SelectionRule,
           typename OpType >
inline typename GenEigsSolver<Scalar, SelectionRule, OpType>::ComplexMatrix GenEigsSolver<Scalar, SelectionRule, OpType>::eigenvectors(int nvec)
{
    int nconv = std::count(ritz_conv.begin(), ritz_conv.end(), true);
    nvec = std::min(nvec, nconv);
    ComplexMatrix res(dim_n, nvec);

    if(!nvec)
        return res;

    ComplexMatrix ritz_vec_conv(ncv, nvec);
    int j = 0;
    for(int i = 0; i < nev && j < nvec; i++)
    {
        if(ritz_conv[i])
        {
            ritz_vec_conv.col(j) = ritz_vec.col(i);
            j++;
        }
    }

    res = fac_V * ritz_vec_conv;

    return res;
}
