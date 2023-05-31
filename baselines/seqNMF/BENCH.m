for p_drop = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
% for p_drop = [0.2]
    for jitter_std = [5, 10, 20, 30, 40, 50]
        runid = 0
        
        tic

        load(sprintf('data/matlab_dataset_%.1f_%i.mat', p_drop, jitter_std));
        X_ = double(X_);

        number_of_seqences = 1;
        Nneurons = size(X_,1);


        K = 1;
        L = 100;
        lambda =.005;
        maxiter = 100;

        shg; clf

        msg = sprintf('Running seqNMF on the dataset with %i neurons, %i timesteps', Nneurons, i);
        display(msg)

        [W,H] = seqNMF(X_,'K',K, 'L', L,'lambda', lambda, 'maxiter', maxiter);


        elapsed_s = toc;
        display(sprintf('elapsed %f seconds', elapsed_s));

        % fileID = fopen('seqNMF_runtimes_DOUBLE_0.csv', 'a');
        % fprintf(fileID, '%s,%s,%s\n', string(Nneurons), string(i), string(elapsed_s));
        % fclose(fileID);

        save(sprintf('results_%.1f_%i_%i.mat', p_drop, jitter_std, runid), 'H')
    end
end
