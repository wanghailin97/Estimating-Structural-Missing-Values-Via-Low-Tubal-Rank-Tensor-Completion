function [X,obj,err,iter] = lrtc_str(M,omega,opts)

% Low-Rank Tensor Completion (LRTC) for Structured Observations based on Tensor Nuclear Norm (TNN) problem by M-ADMM
%
% min_X ||X||_* + lambda*||P_Omega^c(X)||_1, s.t. P_Omega(X) = P_Omega(M)
%
% ---------------------------------------------
% Input:
%       M       -    d1*d2*d3 tensor
%       omega   -    index of the observed entries
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       X       -    d1*d2*d3 tensor
%       err     -    residual
%       obj     -    objective function value
%       iter    -    number of iterations
%
% version 1.0 - 01/07/2020
%
% Written by Hailin Wang (wanghailin97@163.com)
% 
% References: Estimating Structural Missing Values Via Low-Tubal-Rank Tensor Completion, ICASSP, 2020

lambda = 1e-2; % need to be fine-tuned for better berformance 
tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;
DEBUG = 0;


if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'lambda');      lambda = opts.lambda;        end
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

dim = size(M);
omega_c = setdiff((1:prod(dim)),omega);

X = zeros(dim);
X(omega) = M(omega);
S = zeros(dim);
Y = S;

for iter = 1 : max_iter
    Xk = X;
    Sk = S;
    % update X
    [X,tnnX] = prox_tnn(M+S-Y/mu,1/mu); 
    % update S
    temp = X-M+Y/mu;
    S(omega_c) = prox_l1(temp(omega_c),lambda/mu);
 
    dY = X-S-M;    
    chgX = max(abs(Xk(:)-X(:)));
    chgS = max(abs(Sk(:)-S(:)));
    chg = max([chgX chgS max(abs(dY(:)))]);
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            obj = tnnX+lambda*norm(S(:),1);
            err = norm(dY(:));
            disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err)]); 
        end
    end
    
    if chg < tol
        break;
    end 
    Y = Y + mu*dY;
    mu = min(rho*mu,max_mu);    
end

if DEBUG
    obj = tnnX+lambda*norm(S(:),1);
    err = norm(dY(:));
    disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
        ', obj=' num2str(obj) ', err=' num2str(err)]);
end

end

 