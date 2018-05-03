% %% generate data set
% clc; clear; close all;
% m1=500;% number of every set
% m2=500;
% m=m1+m2;
% mu=[0 0 0]';
% sigma=2*[3 2 1;2 3 2;1 2 3 ];
% X1=mvnrnd(mu,sigma,m1);
% scatter3(X1(:,1), X1(:,2), X1(:,3),[],'b'); hold on;
% 
% mu1=[-5 5 0]';
% sigma1=2*[3 2 1;2 3 2;1 2 3 ];
% X2=mvnrnd(mu1,sigma1,m2);
% scatter3(X2(:,1), X2(:,2), X2(:,3),[],'r'); hold on;
% 
% %% LABEL and reshape the data set
% X=zeros(m,1+3+1);
% X(:,1)=ones(m,1); %extend the vector to n+1 d
% X(1:m1,2:4)=X1; X(1:m1,5)=ones(m1,1); %x1 for positive label
% X(m1+1:m,2:4)=X2; %x2 for negative label
% % rearrange the data randomly
% rand_index = randperm( m );   
% X = X( rand_index,: );   
% 
% save('data1000_2_5_doublesigma','X');

%% train perceptron
 %close all; clear; clc;
load('data5002_30_4','X');
max_iter=100;
yita=0.1;
% [ w, cost,num_misclass,iter ] = single_sample_perceptron( X, max_iter ,yita );
[ w, cost,num_misclass ,iter ] = batch_perceptron( X, max_iter ,yita );
%% draw scatter
% scatter3(X( X(:,5)==0,2), X( X(:,5)==0,3), X( X(:,5)==0,4),[],'b'); hold on;
% scatter3(X( X(:,5)==1,2), X( X(:,5)==1,3), X( X(:,5)==1,4),[],'r'); hold on;

%% draw the dividing surface
%  x=floor(min(X(:,2))):1: ceil(max(X(:,2)));
%  y=floor(min(X(:,3))):1: ceil(max(X(:,3)));
%  [x,y]=meshgrid(x,y);
%  z=-(w(2)*x+w(3)*y+w(1))./w(4);
%  surf(x,y,z); zlim( [ floor(min(X(:,3))) , ceil( max(X(:,3)) )] );
 
 plot(cost,'b'); xlim([2 iter+1]); xlabel('iteration times'); ylabel('cost function value');hold on;%title('separability=0.625');
% hold on; plot(num_misclass,'r--o');
% % shading interp
% % set(gca,'ZLim',[-6 6])
% % axis equal
% alpha(0.6)
% ones(16,21)*


