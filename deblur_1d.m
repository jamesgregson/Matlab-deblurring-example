function [deblurred] = deblur_1d( kernel, intrinsic, blurred, reg, regweight )
    % define default parameters for the
    % regularizer options
    if (nargin < 4) || isempty( reg )
        reg = 'Tikhonov'; 
    end
    if (nargin < 5) || isempty( regweight )
        regweight = 0.0001;
    end

    % find the size of the input
    N = size( blurred, 1 );
    
    % create a new kernel at the same size as
    % the input blurry image
    Nk = size( kernel, 1 );
    K = zeros( N, 1 );
    K( N/2-Nk/2 : N/2+Nk/2-1 ) = kernel;
      
    % generate a dirac kernel
    is_L1 = false;
    is_L2 = true;
    H = zeros( N, 1 );
    if strcmp(reg,'Tikhonov')
        H(N/2) = 1.0;
    end
    if strcmp(reg,'Laplace')
        H(N/2-1) =  1.0;
        H(N/2+0) = -2.0;
        H(N/2+1) =  1.0;
    end
    if strcmp(reg,'Biharmonic')
        H(N/2-2) =  1.0;
        H(N/2-1) = -4.0;
        H(N/2+0) =  6.0;
        H(N/2+1) = -4.0;
        H(N/2+2) =  1.0;
    end
    
    if strcmp(reg,'L1')
        is_L2 = false;
        is_L1 = true;
        H(N/2) = 1.0;
    end
    
    if strcmp(reg,'TV')
        is_L2 = false;
        is_L1 = true;
        H(N/2+0) = -1.0;
        H(N/2+1) =  1.0;
    end

    if strcmp(reg,'SparseCurvature')
        is_L2 = false;
        is_L1 = true;
        H(N/2-1) =  1.0;
        H(N/2+0) = -2.0;
        H(N/2+1) =  1.0;
    end
    
    if strcmp(reg,'HyperLaplacian')
        is_L2 = false;
        is_L1 = false;
        %H(N/2-1) = -1.0;
        %H(N/2+0) =  2.0;
        %H(N/2+1) = -1.0;
        H(N/2+0) =  1.0;
        H(N/2+1) = -1.0;

    end

    if is_L2
        % solve for the deblurred image for
        % the L2 priors (Tikhonov, Laplace &
        % Biharmonic) using a Fourier method
    
        % compute the Fourier transforms of the
        % blur kernel and input blurry image
        fK = fft( K );
        fB = fft( blurred );

        % Fourier transforms of the kernels
        fH = fft( H );    

        % compute the Fourier transform of the
        % deblurred image
        fI = ( conj(fK).*fB./(conj(fK).*fK + fH*regweight) );
        I = real(ifftshift(ifft(fI)));

        % set the deblurred output
        deblurred = I;        
    else 
        % using an L1 prior, solve using ADMM
        
        rho = 10.0;
        
        % initialize the solution to the blurred image
        % the langrange multipliers to zero and the 
        % splitting variable to the convolution of the 
        % prior matrix with the solution
        I = intrinsic;
        u = zeros( N, 1 );
        z = conv( I, H, 'same' );
        
        % compute the Fourier transforms of the
        % blur kernel and input blurry image
        fK = fft( K );
        fB = fft( blurred );
        fH = fft( H );
        
        for j=1:1000,
            % solve for the primary degrees of freedom
            ft = fft( z-u );
            fI = ( conj(fK).*fB + regweight*conj(fH).*(ft) )./( conj(fK).*fK + regweight*conj(fH).*fH );
            I = real(ifftshift(ifft(fI)));
            
            tmp = conv( I, H, 'same' );
            if is_L1
                % solve for the splitting variables using shrinkage. this is
                % the proximal operator for a generalized Lasso problem
                kappa = regweight/rho;
                z = max(tmp+u-kappa,0) - max(-tmp-u-kappa,0); %sign(tmp+u).*max(0,abs(tmp+u)-kappa);
        
                % update the lagrange multipliers
                u = u + tmp - z;
            else
                % solve for the splitting variable using hyper-laplacian
                v = tmp;
                for k=1:N,
                    
                   % this solver is based on finding the roots of a cubic
                   % equation whose solutions represent a non-convex
                   % version of the proximal operator.  we first generate
                   % the polynomial and store it in p, then compute the 
                   % roots using matlab's roots() function.  The remaining
                   % code then looks for the best real-root between 
                   % 0 and v(k), comparing this to the objective value
                   % obtained by setting z(k) == 0
                   p = [ 1, -2*v(k), v(k)^2, - sign(v(k))/(4.0*rho) ];
                   r = roots(p);
                   best = 0;
                   bestval = (rho/2.0)*v(k)^2;
                   for q=1:size(r,1),
                       if abs(imag(r(q))) < 1e-6
                          test = min(v(k),max(0,real(r(q))));
                          if test > 0 && test <= v(k) 
                              testval = regweight*test^0.5 + (rho/2.0)*(v(k)-test)^2;
                              if( testval < bestval )
                                  best=test;
                                  bestval=testval;
                              end
                          end
                       end
                   end
                   z(k) = best;
                end
                % update the lagrange multipliers
                u = u + tmp - z;
            end
        end
        
        deblurred = I;
    end
    


end