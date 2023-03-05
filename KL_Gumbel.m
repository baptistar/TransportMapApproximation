clear; close all; clc
addpath(genpath('~/YourPathToATM/src'))

% define reference
ref = Normal();

% define parameters
mu = 0;
beta = 1;

% define true target density and map
sample_nu   = @(N) -1*evrnd(0,beta,N,1) + mu;
pdf_true    = @(x) evpdf(-1*(x-mu),0,beta);
Tmap        = @(x) sqrt(2)*erfinv(2*(1-evcdf(-1*(x-mu),0,beta)) - 1);
Tmap_inv    = @(x) -1*evinv((1-erf(x/sqrt(2)))/2,0,beta) + mu;

%% Check convergence in KL

% set sample sizes
Ntrain = 1000;

% set samples and weights
Xtrain = sample_nu(Ntrain);
Wtrain = ones(Ntrain,1)/Ntrain;

% eval true map
xx = linspace(-3,10,1000);
TrueMap = Tmap(xx);

% plot true map
figure()
hold on
plot(xx, TrueMap, 'Linewidth',5,'DisplayName', 'True Map')

% plot approximate map
orders = [1,5,10];
for i=1:length(orders)
    % approximate map
    basis = ProbabilistHermiteFunction();
    coeff = zeros(orders(i)+1,1); coeff(1) = 1;
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'off');
    obj = @(a) KL_divergence(a, ref, basis, Xtrain, Wtrain);
    [a_opt, ~, exit_flag] = fminunc(obj, coeff, options);
    % eval approximate map
    [ApproxMap, ~] = evaluate_map(a_opt, basis, xx.');
    plot(xx, ApproxMap, 'DisplayName', ['Degree $' num2str(orders(i)) '$'])
end
legend('show')
ylim([-3.5,3.5])
xlim([-3,10])
hold off

%% Run optimization for each order

% set sample sizes
Ntrain = 10000;
Ntest  = 100000;

% set orders
orders = 1:20;

% define arrays to store KL and L2 error in the map along with error bars
kldivergence            = zeros(length(orders), 2);
kldivergence_err        = zeros(length(orders), 2);
l2norm_map              = zeros(length(orders), 2);
l2norm_map_err          = zeros(length(orders), 2);

% generate training samples
Xtrain = sample_nu(Ntrain);
Wtrain = ones(Ntrain,1)/Ntrain;

% generate test samples
Xtest = sample_nu(Ntest);
Wtest = ones(Ntest,1)/Ntest;

% evaluate true map
target_KR = Tmap(Xtest);

% evaluate true PDF
target_logpdf = log(pdf_true(Xtest));

for l=1:length(orders)
    fprintf('N = %d, order = %d\n', Ntrain, orders(l))        

    % set map order, basis, and coefficients
    basis = ProbabilistHermiteFunction();
    coeff = zeros(orders(l)+1,1);
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'off');
    obj = @(a) KL_divergence(a, ref, basis, Xtrain, Wtrain);
    [coeff_opt, ~, exit_flag] = fminunc(obj, coeff, options);

    % interpolate log-pdf
    [NLL, ~] = negative_log_likelihood(coeff_opt, ref, basis, Xtest);
    log_pdf_diff = target_logpdf + NLL;

    % evaluate kl-divergence and error
    kldivergence(l,1) = sum(log_pdf_diff .* Wtest);
    kldivergence(l,2) = 1.96*std(Ntest * log_pdf_diff .* Wtest)/sqrt(length(Xtest));

    % evaluate true and approximate map
    [approx_map, ~] = evaluate_map(coeff_opt, basis, Xtest);

    % evaluate weighted l2 norm of map
    map_diff = (target_KR - approx_map).^2; map_diff(isinf(map_diff)) = [];
    l2norm_map(l,1) = sqrt(sum(map_diff .* Wtest));
    l2norm_map(l,2) = 1.96*std(Ntest * map_diff .* Wtest)/sqrt(length(Xtest));

end

%% plot results

figure()
hold on
errorbar(orders, kldivergence(:,1), kldivergence(:,2), '-o', ...
    'LineWidth', 2, 'DisplayName', '$D_{KL}(\nu||\widehat{T}^{\sharp}\eta)$')
errorbar(orders, l2norm_map(:,1), l2norm_map(:,2), '-o', ...
    'LineWidth', 2, 'DisplayName', '$\|T - \widehat{T}\|_{L_2(\eta)}$')
xlabel('Polynomial degree, n')
set(gca,'YScale','log')
legend('show')
grid on
hold off

%% define objective and gradient

function [x,w] = rescale_CCpts(a, b, x, w)
	% Rescale the quadratures nodes x and weights w for
	% integrating a function with respect to [a,b]
    assert(all(size(a) == size(b)))
    assert(size(x,2) == 1)
    assert(size(w,2) == 1)
	x = 0.5*(b + a) + 0.5*(b - a)*x.';
	w = 0.5*(b - a).*w.';
end

function [T, dxT, dcT, dcdxdT] = evaluate_map(coeff, basis, x)

    % define rectifier
    g = Rectifier('softplus');

    % define the number of quadrature points
    Nquad = 200;
    
    % extract and rescale CC nodes and weights to domain [0,x]
    [xcc, wcc] = clenshaw_curtis(Nquad);
    [xcc, wcc] = rescale_CCpts(zeros(size(x,1),1), x, xcc, wcc);

    % evaluate f at zero, x, and quadrature points
    Psi0      = basis.grad_vandermonde(zeros(size(x,1),1), length(coeff)-1, 0, true);
    dxPsi_x   = basis.grad_vandermonde(x, length(coeff)-1, 1, true);
    dxPsi_xcc = basis.grad_vandermonde(xcc(:), length(coeff)-1, 1, true);
        
    % evaluate f(x) and f'(x)
    f0        = Psi0 * coeff;
    dxf_x     = dxPsi_x * coeff;
    dxf_xcc   = dxPsi_xcc * coeff;

    % evaluate map and derivative
    dxf_xcc_r = reshape(dxf_xcc, size(xcc,1), size(xcc,2));
    T   = f0 + sum(g.evaluate(dxf_xcc_r) .* wcc, 2);
    dxT = g.evaluate(dxf_x);
 
    if nargout > 2
        
        % evaluate gradients of map with respect to coeffs
        dcdxf_xcc = g.grad_x(dxf_xcc) .* dxPsi_xcc;
        dcdxf_xcc_r = reshape(dcdxf_xcc, size(xcc,1), size(xcc,2), length(coeff));
        dcT = Psi0 + squeeze(sum(dcdxf_xcc_r .* reshape(wcc, size(wcc,1), size(wcc,2), 1), 2));

        % evaluate gradient of dxT with respect to coeffs
        dcdxdT = g.grad_x(dxf_x) .* dxPsi_x;
        
    end

end

function [L,dcL] = negative_log_likelihood(coeff, ref, basis, x)
    
    % evaluate map
    if nargout == 1
        [Sx, dxdS] = evaluate_map(coeff, basis, x);
    elseif nargout == 2
        [Sx, dxdS, dcS, dcdxdS] = evaluate_map(coeff, basis, x);
    end
    
    % add small regularization term to map
    delta = 1e-12;
    Sx = Sx + delta*x;
    dxdS = dxdS + delta;
    
    % evaluate log_pi(x)
    L = ref.log_pdf(Sx) + log(dxdS);
    L = -1 * L;
    
    % evaluate gradient \nabla_c log_pi(x)
    if nargout == 2
        dcL = ref.grad_x_log_pdf(Sx) .* dcS + dcdxdS ./ dxdS;
        dcL = -1 * dcL;
    end

end

function [L,dcL] = KL_divergence(coeff, ref, basis, x, w)
    
    % evaluate negative log likelihood
    if nargout == 1
        L = negative_log_likelihood(coeff, ref, basis, x);
    elseif nargout == 2
        [L,dcL] = negative_log_likelihood(coeff, ref, basis, x);
    end
   
    % take average of likelihood and gradients
    L = sum(L .* w);
    if nargout == 2
        dcL = sum(dcL .* w);
    end

end

% -- END OF FILE --
