tic

load('HVC.mat');
X_ = double(X_);

T = size(X_,2); % length of data to generate
Nneurons = size(X_,1);


K = 2;
L = 100;
lambda =.005;
maxiter = 100;

shg; clf

msg = sprintf('Running seqNMF on the dataset with %i neurons, %i timesteps', Nneurons, i);
display(msg)

[W,H] = seqNMF(X_,'K',K, 'L', L,'lambda', lambda, 'maxiter', maxiter);


elapsed_s = toc;

fileID = fopen('seqNMF_HVC.csv', 'a');
fprintf(fileID, '%s,%s,%s\n', string(Nneurons), string(i), string(elapsed_s));
fclose(fileID);
