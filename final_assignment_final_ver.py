# -*- coding: utf-8 -*-
"""

PHYS20161 - Final assignment: Nuclear decay

This script:
Reads and validates data files and combines them.
Performs a minimized chi squared fit and returns fit parameters
which are the decay constants for Rubidium and Strontium.
Calculates the half life of both elements.
Returns uncertainties for both half life and decay constant.
Calculates the reduced chi square of the fit.
Creates and saves a Activity vs time plot, residuals and a contour plot
of the chi squared values varying with decay constants.
Finds the activity and it's uncertainty at any time using
the fitted function.

@author: e57788rc
Last updated: 14/12/2022
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy import stats
from sympy import diff,symbols,exp

NUMBER_ATOMS = 6.0221408e+23*10**(-6)


def initial_files():
    '''
    Function stores default file names and also asks user if they want to use
    other files instead of the default files.

    Args:
        None
    Returns:
        file_names(list)
        None
    '''
    while True:
        print('Use custom files?')
        user_input = input('Enter yes or no:')
        if user_input in ['yes','YES','y','Yes']:
            file_names = input_file_names()
            return file_names
        if user_input in ['no','NO','No','n']:
            file_names = ['Nuclear_data_2.csv','Nuclear_data_1.csv']
            return file_names
        print('Please enter yes or no.')
    return None


def input_file_names():
    '''
    Asks user to input custom file names as a string.

    Args:
        None
    Returns:
        file_names(list)
    '''
    file_names = []
    while True:
        temp_file_name = input('Input file name:')
        file_names.append(temp_file_name)
        while True:
            print('Input another file?')
            input_another = input('Enter yes or no:')
            if input_another in ['no','NO','No','n']:
                return file_names
            if input_another in ['yes','YES','y','Yes']:
                break
            print('Please enter yes or no.')
    return None


def read_file(file_name):
    '''
    Function reads a given file name, creates an numpy array and removes
    bad values (nan, inf, zero uncertainty, negative values) from the array.

    Args:
        file_name(string)
    Returns:
        raw_data(array)
        None
    '''
    data_open = False

    try:
        file = open(file_name,'r',encoding='utf-8')
        data_open = True

    except FileNotFoundError:
        print('Unable to open file.')

    if data_open is True:
        try:
            raw_data = np.genfromtxt(file, delimiter=',', skip_header=0)
            index_nan_values = np.where(np.isnan(raw_data) == True)
            index_inf_values = np.where(np.isinf(raw_data) == True)
            index_negative = np.where(raw_data < 0)
            index_bad_uncertainty = np.where(raw_data[:,2] <= 0)
            index_bad_values = np.hstack((index_nan_values[0],index_inf_values[0],
                                          index_bad_uncertainty[0],index_negative[0]))
        except IndexError:
            #Raises when the file has less than 3 columns
            print('Please use a file that has 3 columns.')
            return None

        #removes duplicate values
        index_bad_values = list(set(index_bad_values))
        raw_data = np.delete(raw_data,index_bad_values, 0)
        raw_data = convert(raw_data)
        file.close()
        return raw_data
    return None


def convert(data):
    '''
    Converts data Tbq to Bq and hours to seconds.

    Args:
        data(array)
    Returns:
        data(array)
    '''
    data[:,0] = data[:,0]*3600
    data[:,1] = data[:,1]*10**(12)
    data[:,2] = data[:,2]*10**(12)
    return data


def function(time,decay_constant_rb,decay_constant_sr):
    '''
    This function stores equation 4 which is the equation
    we are trying to fit. The equation relates decay constant,
    time and activity.

    Args:
        time(float)
        decay_constant_rb(float)
        decay_constant_sr(float)
    Returns:
        The equation relating decay constants, time and activiy
    '''
    #ignores warnings as fmin tries different decay constants which may result in dividebyzero
    warnings.simplefilter('ignore', RuntimeWarning)
    return decay_constant_rb * NUMBER_ATOMS * (decay_constant_sr/
            (decay_constant_rb - decay_constant_sr)*(np.exp(-decay_constant_sr*time)
                                                  - np.exp(-decay_constant_rb*time)))


def remove_outliers(raw_data,decay_constant_rb,decay_constant_sr):
    '''
    This function checks if a data point is too far away from an initial fit
    and removes it as an outlier.

    Args:
        raw_data(array)
        decay_constant_rb(float)
        decay_constant_sr(float)
    Returns:
        data(array)
    '''
    data = np.zeros((0,3))
    for line in raw_data:
        if abs(line[1] - function(line[0],decay_constant_rb,decay_constant_sr)) < 3*line[2]:
            data = np.vstack((data,line))
    return data


def chi_squared(decay_constants,data):
    '''
    This function stores the equation for chi square.

    Args:
        decay_constants(tuple)
        data(array)
    Returns:
        The equation for chi square.
    '''
    decay_constant_rb = decay_constants[0]
    decay_constant_sr = decay_constants[1]
    return np.sum(((data[:,1]-function(data[:,0],decay_constant_rb,decay_constant_sr))/
                   data[:,2])**2)


def minimize(data):
    '''
    This function uses fmin to find the values of decay constants which
    minimizes the value of chi square.
    It also checks for the smaller decay constant which is the Rb
    decay constant.

    Args:
        data(array)
    Returns:
        decay_constant_rb(float)
        decay_constant_sr(float)
    '''
    rb_start = 0.0005
    sr_start = 0.005
    result = fmin(chi_squared, (rb_start,sr_start), args=(data, ), full_output=True, disp=False)
    decay_constants = result[0]

    #since we know Rb has a lower decay constant we can distinguish between them
    if decay_constants[0] < decay_constants[1]:
        decay_constant_rb = decay_constants[0]
        decay_constant_sr = decay_constants[1]
    else:
        decay_constant_rb = decay_constants[1]
        decay_constant_sr = decay_constants[0]

    return decay_constant_rb, decay_constant_sr


def plot_activity_vs_time(decay_constant_rb,decay_constant_sr,data):
    '''
    This function plots the data and the fitted function.
    x-axis is the time in minutes
    y-axis is the activity in TBq

    Args:
        decay_constant_rb(float)
        decay_constant_sr(float)
        data(array)
    Returns:
        None
    '''
    time = np.linspace(0,3600,10000)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))

    axes = fig.add_subplot(111)
    axes.errorbar(data[:, 0]/60, data[:, 1]*10**(-12), yerr=data[:, 2]*10**(-12),
                  fmt='x', color='#0C7489',ls='none',capsize=2,capthick=1)
    axes.plot(time/60,function(time,decay_constant_rb,decay_constant_sr)*10**(-12),color='#040404')

    axes.set_title('Activity vs time of Rb decay')
    axes.set_xlabel('Time in minutes')
    axes.set_ylabel('Activity in TBq')
    axes.legend(['Fitted function', 'Data'])
    plt.savefig('Activity vs time',dpi=200)
    plt.show()


def residual_plot(data,decay_constant_rb,decay_constant_sr):
    '''
    Finds and plots the residuals.

    Args:
        dats(array)
        decay_constant_rb(float)
        decay_constant_sr(float)
    Returns:
        None
    '''
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    residuals = (data[:,1] - function(data[:,0],decay_constant_rb,decay_constant_sr))*10**(-12)
    axes_residuals = fig.add_subplot(313)
    axes_residuals.errorbar(data[:,0]/60, residuals, yerr=data[:,2]*10**(-12),
                            ls='none', color='#BF1363',capsize=1,capthick=1,fmt='x')
    axes_residuals.plot(data[:,0]/60, 0 * data[:,0], color='#191923')
    axes_residuals.set_title('Residuals')
    plt.savefig('Residuals',dpi=200)
    plt.show()


def get_data(file_names):
    '''
    This function combines the arrays given by the read_file
    function into one single array.
    Removes really big outliers.
    Creates an initial fit for the raw data and uses the remove_outliers
    function to remove samller outliers.
    This function returns the final data.

    Args:
        file_names(list)
    Returns:
        filtered_data(array)
    '''
    raw_data = np.zeros((0,3))
    for element in file_names:
        raw_data = np.vstack((raw_data,read_file(element)))

    #Removes really big outliers before doing an initial fit
    zscore = stats.zscore(raw_data[:,1],axis=0,nan_policy='omit')
    find_big_outliers = np.where(abs(zscore)>3)
    raw_data = np.delete(raw_data,find_big_outliers[0],0)

    rb_initial,sr_initial = minimize(raw_data)
    filtered_data = remove_outliers(raw_data,rb_initial,sr_initial)
    return filtered_data


def find_uncertainties(data,decay_constant_rb,decay_constant_sr):
    '''
    Using curve_fit and giving it an initial guess of
    decay constants given by fmin; this function finds the
    uncertainties on the fit parameters.

    Args:
        data(array)
        decay_constant_rb(float)
        decay_constant_sr(float)
    Returns:
        uncertainties(tuple)
        covariance(tuple)
    '''
    try:
        _, pcov = curve_fit(function, data[:,0], data[:,1], sigma=data[:,2],
                        p0=(decay_constant_rb,decay_constant_sr),absolute_sigma=True)
        #first uncertainty belongs to Rb, second belongs to Sr
        uncertainties = (pcov[0,0]**0.5 , pcov[1,1]**0.5)
        covariance = (pcov[0,1] , pcov[1,0])
    except RuntimeError:
        print('Unable to find uncertainties after 600 iterations')
        uncertainties,covariance = (0,0)
    return uncertainties,covariance


def input_time():
    '''
    This function asks the user for a time in minutes and
    validates the user input.

    Args:
        None
    Returns:
        user_input(float)
    '''
    while True:
        user_input = input('Input a time in minutes:')
        try:
            user_input = float(user_input)
        except ValueError:
            print('Please Enter a number')
            continue
        if user_input <= 0:
            print('Please Enter a time greater than 0')
            continue
        break
    return user_input


def mesh_grids(data,decay_constant_rb,decay_constant_sr,uncertainties):
    '''
    This function creates three mesh girds.
    x_mesh is a mesh of Rb values.
    y_mesh is a mesh of Sr values.
    z_mesh is a mesh of chi-squared values.

    Args:
        data(array)
        decay_constant_rb(float)
        decay_constant_sr(float)
    Returns:
        x_mesh(array)
        y_mesh(array)
        z_mesh(array)
    '''
    uncertainty_rb = uncertainties[0]
    uncertainty_sr = uncertainties[1]
    x_vals = np.linspace(decay_constant_rb-3*uncertainty_rb,
                         decay_constant_rb+3*uncertainty_rb,200)
    y_vals = np.linspace(decay_constant_sr-3*uncertainty_sr,
                         decay_constant_sr+3*uncertainty_sr,200)
    x_mesh,y_mesh = np.meshgrid(x_vals,y_vals)

    chi_squared_values = []
    for i in y_vals:
        for j in x_vals:
            temp_chi_squared = chi_squared((i,j), data)
            chi_squared_values.append(temp_chi_squared)
    z_mesh = np.array(chi_squared_values).reshape(len(x_vals),len(y_vals))
    return x_mesh,y_mesh,z_mesh


def contour_plot(mesh,chi_square,decay_constant_rb,decay_constant_sr,uncertainties):
    '''
    Plots the mesh grids into a contour plot of chi squared values
    and how it changes with the decay constant parameters.
    Also plots the ellipses for chi-square + 1 and chi-square + 2.3

    Args:
        mesh(tuple)
        chi_square(float)
        decay_constant_rb(float)
        decay_constant_sr(float)
        uncertainties(tuple)
    Returns:
        None
    '''
    x_mesh=mesh[0]
    y_mesh=mesh[1]
    z_mesh=mesh[2]
    uncertainty_rb = uncertainties[0]
    uncertainty_sr = uncertainties[1]
    plt.style.use('seaborn-whitegrid')
    fig,axes=plt.subplots(1,1)
    mycmap1 = plt.get_cmap('inferno')
    axes.set_xlim(decay_constant_rb-3*uncertainty_rb,decay_constant_rb+3*uncertainty_rb)
    axes.set_ylim(decay_constant_sr-3*uncertainty_sr,decay_constant_sr+3*uncertainty_sr)
    contourf_plot = axes.contourf(x_mesh, y_mesh, z_mesh, cmap=mycmap1,extend='both',
                                  levels=np.linspace(chi_square,chi_square+50,100))
    contour_lines = axes.contour(x_mesh,y_mesh,z_mesh,levels=[chi_square+1,chi_square+2.3],
                         colors='white',linestyles='dashed')
    axes.scatter(decay_constant_rb,decay_constant_sr,color='White')
    fig.colorbar(contourf_plot)
    axes.clabel(contour_lines, inline=1, fontsize=8)
    axes.set_title('Contour plot of chi square values')
    axes.set_xlabel('Decay constant Rb')
    axes.set_ylabel('Decay constant Sr')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig('Contour plot',dpi=200)
    plt.show()


def menu():
    '''
    This function defines a menu that asks the user for
    an input.

    Args:
        None
    Returns:
        user_input(float)
    '''

    print('--------------------Extra--------------------')
    print('Enter 1 for a contour plot')
    print('Enter 2 to find the activity at a different time')
    print('Enter 3 to see results again')
    print('Enter 4 to see residuals')
    print('Enter 5 to end program')
    print('----------------------------------------------')
    try:
        user_input = float(input('Enter your input:'))
    except ValueError:
        print('Please enter a number')
        return None
    return user_input


def uncertainty_on_activity(uncertainties,covariance,decay_constant_rb,decay_constant_sr,time):
    '''
    This function finds the uncertainty on the activity at any time
    with error propagation.
    Using formula: variance = (df/da * sigma_a)^2 + (df/db * sigma_b)^2
                              + 2*df/da*df/db*covariance(a,b)
    Args:
        uncertainties(tuple)
        covariance(tuple)
        decay_constant_rb(float)
        decay_constant_sr(float)
    Returns:
        uncertainty(float)
    '''

    variable_rb, variable_sr = symbols('rb sr')
    activity_function = variable_rb*NUMBER_ATOMS*(variable_sr/(variable_rb - variable_sr)*
                        (exp(-variable_sr*time) - exp(-variable_rb*time)))*10**(-12)
    derivative_rb = diff(activity_function,variable_rb)
    derivative_sr = diff(activity_function,variable_sr)

    derivative_rb = derivative_rb.evalf(subs={variable_rb: decay_constant_rb,
                                              variable_sr: decay_constant_sr})
    derivative_sr = derivative_sr.evalf(subs={variable_rb: decay_constant_rb,
                                              variable_sr: decay_constant_sr})

    variance = ((derivative_rb**2)*uncertainties[0]**2 + (derivative_sr**2)*uncertainties[1]**2 +
                2*derivative_rb*derivative_sr*(covariance[0]))
    uncertainty = (variance)**0.5
    return uncertainty


def main():
    '''
    The main function which runs the other functions.
    This function calculates the parameters, uncertainties,
    half lives, chi square value, reduced chi squared value
    and the activity at 90 minutes.
    Then it prints the results.
    It then shows the user a menu which allows the user
    to create a contour plot and find the activity at any
    time.

    Args:
        None
    Returns:
        None
    '''

    file_names = initial_files()
    try:
        data = get_data(file_names)
    except ValueError:
        return None

    #calculating relevent values
    decay_constant_rb, decay_constant_sr = minimize(data)
    half_life_sr = (np.log(2)/decay_constant_sr)/60
    half_life_rb = (np.log(2)/decay_constant_rb)/60
    uncertainties,covariance = find_uncertainties(data,
                                                  decay_constant_rb, decay_constant_sr)
    uncertainty_half_life_rb = half_life_rb*(uncertainties[0]/decay_constant_rb)
    uncertainty_half_life_sr = half_life_sr*(uncertainties[1]/decay_constant_sr)
    chi_square = chi_squared(minimize(data), data)
    reduced_chi_squared = chi_squared(minimize(data), data)/(len(data)-2)
    activity_at_90_mins = function(90*60,decay_constant_rb,decay_constant_sr)*10**(-12)
    uncertainty_at_90_mins = uncertainty_on_activity(uncertainties,covariance,
                                                     decay_constant_rb,decay_constant_sr, 90*60)
    plot_activity_vs_time(decay_constant_rb, decay_constant_sr, data)

    print('-------------------Results-------------------')
    print(f'Decay constant of Rb is {decay_constant_rb:.3} ± {uncertainties[0]:.6f} per second')
    print(f'Decay constant of Sr is {decay_constant_sr:.3} ± {uncertainties[1]:.5f} per second')
    print(f'Half life of Rb is {half_life_rb:.3} ± {uncertainty_half_life_rb:.1f} minutes')
    print(f'half life of Sr is {half_life_sr:.3} ± {uncertainty_half_life_sr:.2f} minutes')
    print(f'Chi squared value of the fit is {chi_square:.2f}')
    print(f'Degrees of freedom = {len(data)-2}')
    print(f'Reduced chi squared of the fit is {reduced_chi_squared:.2f}')
    print(f'Activity at 90 minutes is {activity_at_90_mins:.3} ± {uncertainty_at_90_mins:.1} TBq')

    while True:
        user_input = menu()
        if user_input == 1:
            meshes = mesh_grids(data,decay_constant_rb,decay_constant_sr,uncertainties)
            contour_plot(meshes, chi_square,decay_constant_rb,decay_constant_sr, uncertainties)
        if user_input == 2:
            inputed_time = input_time()*60
            activity_at_time_t = function(inputed_time,decay_constant_rb,
                                          decay_constant_sr)*10**(-12)
            uncertainty_at_time_t = uncertainty_on_activity(uncertainties,covariance,
                                            decay_constant_rb,decay_constant_sr,inputed_time)
            print('---------------------------------------------')
            print(f'Activity at {inputed_time/60} minutes is',
                  f'{activity_at_time_t:.3} ± {uncertainty_at_time_t:.1}TBq')
            print('---------------------------------------------')
        if user_input == 3:
            print('-------------------Results-------------------')
            print(f'Decay constant of Rb is {decay_constant_rb:.3}',
                  f'± {uncertainties[0]:.6f} per second')
            print(f'Decay constant of Sr is {decay_constant_sr:.3}',
                  f'± {uncertainties[1]:.5f} per second')
            print(f'Half life of Rb is {half_life_rb:.1f}',
                  f'± {uncertainty_half_life_rb:.3} minutes')
            print(f'half life of Sr is {half_life_sr:.2f}',
                  f'± {uncertainty_half_life_sr:.3} minutes')
            print(f'Chi squared value of the fit is {chi_square:.2f}')
            print(f'Degrees of freedom = {len(data)-2}')
            print(f'Reduced chi squared of the fit is {reduced_chi_squared:.2f}')
            print(f'Activity at 90 minutes is {activity_at_90_mins:.3} TBq')
            print('---------------------------------------------')
        if user_input == 4:
            residual_plot(data, decay_constant_rb, decay_constant_sr)
        if user_input == 5:
            print('Thank you')
            return None
        print('Enter a a number 1-5')


if __name__ == '__main__':
    main()
