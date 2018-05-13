%
% Generate Plots
%

FontSize= 20;
LineWidth = 2;


fig_cnn_features = figure('units','normalized','outerposition',[0 0 1 1]);
y = [score_nc;...
     score_nsc;...
     score_knn8;...
     score_linreg;...
     score_rbfreg;...
     score_svm;...
     score_mlp]*100;
c = {'NC','NSC','KNN','LIN','RBF','SVM','MLP'};
bar(y)
barvalues;
ylim([0 110]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on


fig_sift_features = figure('units','normalized','outerposition',[0 0 1 1]);
y = [score_nc_sift;...
     score_nsc_sift;...
     score_knn_sift12;...
     score_linreg_sift;...
     score_rbfreg_sift;...
     score_svm_sift;...
     score_mlp_sift]*100;
c = {'NC','NSC','KNN','LIN','RBF','SVM','MLP'};
bar(y)
barvalues;
ylim([0 110]);
set(gca,'xticklabel',c)
set(gca,'FontSize',FontSize);
ylabel('Success Rate [%]','FontSize',FontSize)
grid on



%%
mkdir('plots')
print(fig_cnn_features, 'plots/cnn_features_success','-depsc');
print(fig_sift_features, 'plots/sift_features_success','-depsc');
