function  v = FADE( At,K )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
f_th = 10;  % Cut at 10Hz
f_s  = 120;  % Sampling frequency
%K = 500; % Desired dimensionality
J = 59;
[row,column] = size(At);

At = smoothdata(At,'gaussian' );
%span = 3;
%At = smooth(At,span);
Aw = fft(At);
k = ceil(row/12);
Aw_n = Aw(1:k+1,:);
Aw_n = abs(Aw_n);
w_end = 2*pi*f_th;
x = 0:2*pi*f_th/(size(Aw_n, 1)-1):w_end;
xq = 0:w_end/(K-1):w_end;
Aw_nn = zeros(K, size(Aw_n, 2));
    for idx_innen = 1:column
        v = Aw_n(:, idx_innen);
        Aw_nn(:, idx_innen) = interp1(x,v,xq);
    end
%Aw_nn = resample(Aw_n,K,k+1);
Vw = pca(Aw_nn);
v = Vw(:,1);

end

