clear all;
close all;

% set the number of points in the blurred image
% and in the kernel
Nb = 128;
Nk = 64;

% generate the kernel as a Gaussian blur
sigma = 4;
K = exp( -linspace(-Nk/2, Nk/2, Nk).^2/sigma^2 )';
K = K / sum(sum(K));

% generate a ground-truth signal
O = zeros( Nb, 1 );
O( Nb/4 : 2*Nb/4 ) = 1.0;
for j=Nb/2:3*Nb/4,
   O( j ) = 1.0 - j/(Nb/2); 
end

% generate a blurred image along 
% and corrupt it with noise
B = conv( O, K, 'same' ) + 0.025*randn(Nb,1);

% deblur the blurred image using no
% prior, the L2 norm of the solution,
% the L2 norm of the gradient and the
% L2 norm of the curvature
gamma = 0.5;
na = deblur_1d( K, B, B, 'Tikhonov',   0.0 );
a2 = deblur_1d( K, B, B, 'Tikhonov',   0.01 );
b2 = deblur_1d( K, B, B, 'Laplace',    0.1 );
c2 = deblur_1d( K, B, B, 'Biharmonic', gamma );

% deblur the blurred image using the
% L1 norm of the solution, the L1 norm
% of the gradient and the L1 norm of
% the curvature
gamma = 0.5;
a1 = deblur_1d( K, B, B, 'L1', 0.01 );
b1 = deblur_1d( K, B, B, 'TV', gamma );
c1 = deblur_1d( K, B, B, 'SparseCurvature', gamma );


% regularize the image using the hyper-Laplacian
% regularizer with alpha=0.5, start from an L1
% solution and then relax to the sparser solution
gamma = 0.008;
%tv = deblur_1d( K, B, B, 'TV', 0.5 );
hl = deblur_1d( K, B, B, 'HyperLaplacian', gamma );

% plot the naively deblurred image
h = figure
hold on
plot( O,  'k', 'LineWidth', 2 );
plot( B,  'r', 'LineWidth', 2 );
plot( na, 'b', 'LineWidth', 2 );
axis off
saveas( h, 'naive_deblur', 'png' );

% plot the Tikhonov-regularized solution
h = figure
hold on
plot( O,  'k', 'LineWidth', 2 );
plot( B,  'r', 'LineWidth', 2 );
plot( a2, 'b', 'LineWidth', 2 );
axis off
saveas( h, 'Tikhonov_deblur', 'png' );

% plot the Laplace regularized solution
h = figure
hold on
plot( O,   'k', 'LineWidth', 2 );
plot( B,   'r', 'LineWidth', 2 );
plot( b2,  'b', 'LineWidth', 2 );
axis off
saveas( h, 'Laplace_deblur', 'png' );

% plot the Biharmonically regularized solution
h = figure
hold on
plot( O,   'k', 'LineWidth', 2 );
plot( B,   'r', 'LineWidth', 2 );
plot( c2,  'b', 'LineWidth', 2 );
axis off
saveas( h, 'Biharmonic_deblur', 'png' );

% plot the L1 regularized solution
h = figure
hold on
plot( O,  'k', 'LineWidth', 2 );
plot( B,  'r', 'LineWidth', 2 );
plot( a1, 'b', 'LineWidth', 2 );
axis off
saveas( h, 'L1_deblur', 'png' );

% plot the total-variation regularized solution
h = figure
hold on
plot( O,   'k', 'LineWidth', 2 );
plot( B,   'r', 'LineWidth', 2 );
plot( b1,  'b', 'LineWidth', 2 );
axis off
saveas( h, 'TV_deblur', 'png' );

% plot the sparse-curvature regularized solution
h = figure
hold on
plot( O,   'k', 'LineWidth', 2 );
plot( B,   'r', 'LineWidth', 2 );
plot( c1,  'b', 'LineWidth', 2 );
axis off
saveas( h, 'SparseCurvature_deblur', 'png' )
%}
% plot the hyper-Laplacian solution
h = figure;
hold on
plot( O,  'k', 'LineWidth', 2 );
plot( B,  'r', 'LineWidth', 2 );
plot( hl, 'b', 'LineWidth', 2 );
axis off
saveas( h, 'hyperlaplacian_deblur', 'png' );

