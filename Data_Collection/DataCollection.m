clc
clear
close all

%% Data Collection and Plot
file_name = input('The Name of this Trial in the format GaitName_TrialNum\n', 's');
% initialize collection result with pre-defined num of points 
field_names = 'Voltage_01 Voltage_02 Acc_x Acc_y Acc_z Gyro_x Gyro_y Gyro_z Time';
num_data = 5000;
init_store = zeros(1,num_data);
num_fields = length(strsplit(field_names, ' '));
data_counter = 0;
fileID = fopen([file_name, '.txt'], 'w');
% info about this data file 
fprintf(fileID, [field_names,'\n']);
fprintf(fileID, num2str(num_data));
fprintf(fileID, '\n');
%Define port as 'S' and open port
s=serial('COM4');
s.Baudrate = 250000;
fopen(s);
while length(strsplit(fscanf(s), ' ')) ~= num_fields
    continue 
end

disp('Start Collecting Data...')
while data_counter < num_data
    current_line = fscanf(s);
    fprintf(fileID, current_line);
    data_counter = data_counter + 1;
    % some info about progress
    if data_counter == floor(num_data/3)
        disp('1/3 Data Collected...')
    elseif data_counter == floor(num_data/2)
            disp('1/2 Data Collected...')
    elseif data_counter == floor(num_data/3) * 2
            disp('2/3 Data Collected...')
    end
end
disp('Finishes Collecting Data')
fclose(fileID);
% Close port for other program
fclose(s);
delete(s);