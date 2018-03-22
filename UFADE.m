function  v = UFADE( At,K )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
f_th = 10;  % Cut at 10Hz
f_s  = 120;  % Sampling frequency
J = 59;
[row,column] = size(At);

At = smoothdata(At,'gaussian' );
Aw = fft(At);
k = floor(row/12);
Aw_n = Aw(1:k+1,:);
Aw_n = abs(Aw_n);

%for i=1:J
%    Aw_nn(:,i) = resample(Aw_n(:,i),K,k);
%end
%Aw_nn = resample(Aw_n,K,k+1);
w_end = 2*pi*f_th;
x = 0:2*pi*f_th/(size(Aw_n, 1)-1):w_end;
xq = 0:w_end/(K-1):w_end;
Aw_nn = zeros(K, size(Aw_n, 2));
for idx_innen = 1:column
    v = Aw_n(:, idx_innen);
    Aw_nn(:, idx_innen) = interp1(x,v,xq);
end
v = reshape(Aw_nn,[K*J,1]);

end

