%%%%%%%%%%%%%%%%%%%%%%%%
% This quickstart.m confirms all functions working properly.
% It also shows how to use each function in the toolbox.
%%%%%%%%%%%%%%%%%%%%%%%%

%%% Parameters
ALPHA = 1e-2;
BETA = 1e-2;
OMEGA = 0.05;
LAMBDA = 0.5;
TD = 0.2;
TW = 0.2;
TK = 0.5;
N = 50;
M = 1;
SEED = 1;
OUTPUT = 1;


%%% load data
fprintf(1, 'nips\n*********************\n');
load ../datasets/nips_wd
%load ../datasets/nips_ad
%load ../datasets/nips_dd

train=nips_wd(:,1:1690);
train=sparse(train);
test=nips_wd(:,1691:end);
test=sparse(test);
fprintf(1, 'VB\n*********************\n');
J = 50;
ALPHA = 50/J
BETA = 1/size(train,1)
tic
% [phi, theta, mu] = VBtrain(train, J, N, M, ALPHA, BETA, SEED, OUTPUT); 
% [theta, mu] = VBpredict(test, phi, N, M, ALPHA, BETA, SEED, OUTPUT); 


[phi, thet] = LDAVBtrain(train, J, N, M, ALPHA, BETA,  OUTPUT); 
[theta] = LDAVBpredict(test, phi, N, M, ALPHA, BETA,  OUTPUT);


perplexity(test,phi,theta,ALPHA,BETA)
toc
fprintf(1, '\n*********************\n');
J = 20;
ALPHA = 50/J
BETA = 1/size(train,1)
tic
[phi, theta, mu] = VBtrain(train, J, N, M, ALPHA, BETA, SEED, OUTPUT); 
[theta, mu] = VBpredict(test, phi, N, M, ALPHA, BETA, SEED, OUTPUT); 
perplexity(test,phi,theta,ALPHA,BETA)
toc
fprintf(1, '\n*********************\n');
