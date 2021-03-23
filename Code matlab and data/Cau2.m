clear all;
clc; 
%% Load data from XSENS imu
%% Select dataset "i" from provided datasets in pattern1\...:
 
%----- i=0:4 -----%
i = 0;  % Select i
raw_data  =  load(strcat('pattern1\MT_cal_00300827_00',num2str(i),'.log'));
euler_data  =  load(strcat('pattern1\MT_euler_00300827_00',num2str(i),'.log'));

% raw_data  = load('D:\___HOC TAP\CH\Dieu khien toi uu\Thai Nguyen Trung Thanh_1814031\Bai tap 2 _ EKF\pattern\raw(free2).log');
% euler_data  =  load('D:\___HOC TAP\CH\Dieu khien toi uu\Thai Nguyen Trung Thanh_1814031\Bai tap 2 _ EKF\patterneuler(free2).log');
%% Basic information:
N = max(size(euler_data));      % total dynamic steps     
n = 5;                          % number of state
Ts = 0.01;                      % sample rate
tt = euler_data(:,1);           % time stamp
 
%% Data pre-processing:
g_const = 9.81;
acc  = raw_data(:,2:3)/g_const; % [m/s^2]
gyro = raw_data(:,5:7);         % [rad/s]
mag  = raw_data(:,8:10);        % [mgauge]
mag = mag/norm(mag)*10^(-3);    % [gauge]
 
actual_value = zeros(N,2);              % Actual value
actual_value(:,1) = euler_data(:,3);    % theta [o]     
actual_value(:,2) = euler_data(:,2);    % phi [o]
   
%% Std of process:
% Choose q1 for the best estimation
switch i
    case 0
        q1 = 5;
    case 1
        q1 = 0.5;
    case 2
        q1 = 5;
    case 3
        q1 = 30;
    case 4
        q1 = 1;
    % General case:
    otherwise
        q1 = 10000;
end
Q = diag([0,0,q1,q1,q1]);  
L = diag([0,0,1,1,1])*Ts;
 
%% Std of measurement:
% Choose r1, r2, r3 for the best estimation
switch i
    case 0
        r1 = 0.1;
        r2 = 0.0001;
        r3 = 0.01;
    case 1
        r1 = 0.0006;
        r2 = 0.01;
        r3 = 0.02;
    case 2
        r1 = 0.001;
        r2 = 0.002;
        r3 = 0.01;
    case 3
        r1 = 0.01;
        r2 = 0.01;
        r3 = 0.02;
    case 4
        r1 = 0.00005;
        r2 = 0.0007;
        r3 = 0.01;
    % General case:
    otherwise
        r1 = 0.001;
        r2 = 0.001;
        r3 = 0.001;
end
R = diag([r1,r2,r3,r3,r3]); 
M = eye(5);
 
%% Allocate memory for estimation values:
estimate_value = zeros(N,n);      
 
%% Initialize values:
P_minus = zeros(n,n);       % Initialize P_minus
P = zeros(n,n);             % Initialize P
% Initialize x_hat_minus and x_hat for the best estimation
switch i
    case 0
        x_hat_minus = [-44.62*pi/180; -39.94*pi/180; -0.014954; 0.000145; 0.016868];
        x_hat = x_hat_minus;
    case 1
        x_hat_minus = [1.21*pi/180; 18.19*pi/180; -0.003714; -0.000136; -0.018198];
        x_hat = x_hat_minus;
    case 2
        x_hat_minus = [1.64*pi/180; -1.22*pi/1808; -0.022062; -0.001835; 0.016377];
        x_hat = x_hat_minus;
    case 3
        x_hat_minus = [-10.81*pi/180; 4.46*pi/180; -0.004984; 0.060824; 0.000847];
        x_hat = x_hat_minus;
    case 4
        x_hat_minus = [-16.21*pi/180; 27.80*pi/180; -0.004858; -0.009552; -0.003057];
        x_hat = x_hat_minus;        
    % General case:
    otherwise
        x_hat_minus = zeros(1,n);   
        x_hat = zeros(1,n);         
end
 
%% Kalman filter loop:
for k=1:N-1   
    %% (value substition for the next step)
    theta = x_hat_minus(1);
    phi = x_hat_minus(2);
    omega_x = x_hat_minus(3);
    omega_y = x_hat_minus(4);
    omega_z = x_hat_minus(5);       
    %% Compute Jacobian matrix of g(u_t,x_t_1):  
    Jg12 = - Ts*omega_y*sin(phi) - Ts*omega_z*cos(phi);
    Jg14 = Ts*cos(phi);
    Jg15 = -Ts*sin(phi);
    Jg21 = Ts*omega_z*cos(phi)*(tan(theta)^2 + 1) + Ts*omega_y*sin(phi)*(tan(theta)^2 + 1);
    Jg22 = Ts*omega_y*cos(phi)*tan(theta) - Ts*omega_z*sin(phi)*tan(theta) + 1;
    Jg24 = Ts*sin(phi)*tan(theta);
    Jg25 = Ts*cos(phi)*tan(theta);
    
    Jg = [1, Jg12 , 0 , Jg14 , Jg15;...
       Jg21 , Jg22, Ts , Jg24 , Jg25;...
       0 , 0 , 1 , 0 , 0;...
       0 , 0 , 0 , 1 , 0;...
       0 , 0 , 0 , 0 , 1];    
    
    %% (value substition for the next step)
    theta = x_hat(1);
    phi = x_hat(2);
    omega_x = x_hat(3);
    omega_y = x_hat(4);
    omega_z = x_hat(5);  
    
    g11 = theta - Ts*omega_z*sin(phi) + Ts*omega_y*cos(phi);
    g21 = phi + Ts*omega_x + Ts*omega_z*cos(phi)*tan(theta) + Ts*omega_y*sin(phi)*tan(theta);
    g31 = omega_x;
    g41 = omega_y;
    g51 = omega_z;
    
    g=[g11; g21; g31; g41; g51];
 
    %% Project the error covariance ahead:
    x_hat_minus = g;
    P_minus = Jg*P*Jg' +L*Q*L';
    
    %% (value substition for the next step)
    theta = x_hat_minus(1);
    phi = x_hat_minus(2);
    omega_x = x_hat_minus(3);
    omega_y = x_hat_minus(4);
    omega_z = x_hat_minus(5);  
 
    %% Compute Jacobian matrix of h(x_t): 
    Jh11 = -cos(theta);
    Jh21 = -sin(phi)*sin(theta);
    Jh22 = cos(phi)*cos(theta);   
  
    Jh=[Jh11 , 0 , 0 , 0 , 0;
        Jh21 , Jh22 , 0 , 0 , 0;
        0 , 0 , 1 , 0 , 0;
        0 , 0 , 0 , 1 , 0;
        0 , 0 , 0 , 0 , 1];
    
    %% Compute Kalman gain:
    S = Jh*P_minus*Jh'+M*R*M';
    K = P_minus*Jh'*inv(S); 
    
    %% (value substition for the next step)
    h11 = -sin(theta);
    h21 = sin(phi)*cos(theta);
    h31 = omega_x;
    h41 = omega_y;
    h51 = omega_z;
    
    hx = [h11; h21; h31; h41; h51]; 
    
    %% Update estimate with measurement:    
    z = [acc(k,1); acc(k,2); gyro(k,1); gyro(k,2); gyro(k,3)];
    x_hat = x_hat_minus + K*(z - hx); 
    estimate_value(k,:) = x_hat;   % Store estimate values 
    
    %% Compute error covariance for updated estimate:
    P = (eye(n)-(K*Jh))*P_minus;  
    
    %% Estimate value data post-processing:
    while(estimate_value(k,1)<-pi)
        estimate_value(k,1)=estimate_value(k,1)+2*pi;
    end
    while(estimate_value(k,2)<-pi)
        estimate_value(k,2)=estimate_value(k,2)+2*pi;
    end     
    while(estimate_value(k,1)>pi)
        estimate_value(k,1)=estimate_value(k,1)-2*pi;
    end
    while(estimate_value(k,2)>pi)
        estimate_value(k,2)=estimate_value(k,2)-2*pi;
    end  
end
 
%% Visualize the results
title_labels = {'Estimation of theta angle', 'Estimation of phi angle'};
legend_labels = {{'theta', 'theta estimate'}, {'phi', 'phi estimate'}};
for i = 1:2
    figure(i);
    plot(tt, actual_value(:,i),'r');
    hold on;
    plot(tt, estimate_value(:,i)*180/pi,'b');
    hold off;     
    legend(legend_labels{i});
    plot(tt, abs(estimate_value(:,i)*180/pi-actual_value(:,i)),'g');
    hold on;
    xlabel('Time [s]');
    ylabel('Angle [o]');
    grid on;
    
    %% Fitness evaluation
    fitness = fitnessCalculator(estimate_value(:,i)*180/pi, actual_value(:,i), N);
    RMSE = errorCalculator(estimate_value(:,i)*180/pi, actual_value(:,i), N);
    txt = append('_ - FITNESS = ',num2str(fitness),'%',', RMSE = ', num2str(RMSE), ' [o]');
    title(append(title_labels{i}, txt));
end


 
function [fitness] = fitnessCalculator(estimate, true, N)
mean_value = mean(true(:));
a=0;
b=0;
for k=1:N
    a=a+(true(k)-estimate(k))^2;
    b=b+(true(k)-mean_value)^2;  
end
fitness = (1-a/b)*100;
end
function [RMSE] = errorCalculator(estimate, true, N)
RMSE = 0;
for k=1:N
    RMSE = RMSE + (true(k)-estimate(k))^2;
end
RMSE = sqrt(RMSE/(N));
end