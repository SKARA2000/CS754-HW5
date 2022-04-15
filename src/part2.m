%% Clearing console and variables
clc; clear all;
%% Initial Parameters for solving the part
m = [40, 50, 64, 80, 100, 120];
alpha = 0:0.5:3;
% alpha = 0:1:3;
n = 128;
c = 1;
U = RandOrthMat(n, 43, 1e-6);
rmse = zeros(length(m) ,length(alpha));
loops = 10;
for i=1:length(alpha)
    Lambda = zeros(128, 1);
    for j=1:n
        Lambda(j) = sqrt(j^(-alpha(i)));
    end
    A = U*diag(Lambda); % Because it is the matrix we will be using for generating the random vector x which is normally generated.
    % AA' = Required Covariance Matrix(Sigma_x)
    rng(44);
    for j=1:length(m)
        for k=1:loops
            x = A*randn(n,1);   
            phi = sqrt(1/m(j))*randn(m(j), n);
            y_inter = phi*x;
            total_avg = sum(abs(y_inter), 'all')/(m(j)*n);
            sigma = 0.01*total_avg;
            N = sigma*randn(m(j), 1);
            y_meas = y_inter + N;
            % Using the MAP estimate for x
            reconstructed_x = (inv(phi'*phi + sigma^2*(U*diag(1./(Lambda.^2))*U')))*phi'*y_meas;
            rmse(j, i) = rmse(j, i) + norm(reconstructed_x - x)/sqrt(n);
        end
        rmse(j, i) = rmse(j, i)/loops;
    end
end
figure();
for i=1:length(alpha)
    plot(m, log(rmse(:, i) + 0.01), '-*');
    hold on;
end
grid on;
title('RMSE vs. m');
ylabel('log(RMSE)');
xlabel('Number of measurements(m)');
legend('alpha = 0', 'alpha = 0.5', 'alpha = 1', 'alpha = 1.5', 'alpha = 2', 'alpha = 2.5', 'alpha = 3');
% legend('alpha = 0', 'alpha = 1', 'alpha = 2', 'alpha = 3');
saveas(gcf(), '../images/plot.png');

function M=RandOrthMat(n, seed, tol)
% M = RANDORTHMAT(n)
% generates a random n x n orthogonal real matrix.
%
% M = RANDORTHMAT(n,tol)
% explicitly specifies a thresh value that measures linear dependence
% of a newly formed column with the existing columns. Defaults to 1e-6.
%
% In this version the generated matrix distribution *is* uniform over the manifold
% O(n) w.r.t. the induced R^(n^2) Lebesgue measure, at a slight computational 
% overhead (randn + normalization, as opposed to rand ). 
% 
% (c) Ofek Shilon , 2006.
    if nargin==1
	  tol=1e-6;
    end
    
    M = zeros(n); % prealloc
    rng(seed);
    % gram-schmidt on random column vectors
    
    vi = randn(n,1);  
    % the n-dimensional normal distribution has spherical symmetry, which implies
    % that after normalization the drawn vectors would be uniformly distributed on the
    % n-dimensional unit sphere.
    M(:,1) = vi ./ norm(vi);
    
    for i=2:n
	  nrm = 0;
	  while nrm<tol
		vi = randn(n,1);
		vi = vi -  M(:,1:i-1)  * ( M(:,1:i-1).' * vi )  ;
		nrm = norm(vi);
	  end
	  M(:,i) = vi ./ nrm;
    end %i
        
end  % RandOrthMat