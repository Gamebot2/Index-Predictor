%Initialization
clear; close all; clear;

%Data Extraction stage
data = csvread('data.csv');
row = data(2, :);
fprintf('%.3f\n ', row);