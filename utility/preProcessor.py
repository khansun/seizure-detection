import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import mne 
from termcolor import colored
import re
import warnings
import pyedflib
from scipy import signal
from sklearn.metrics import classification_report, roc_curve, auc

def chb_mit_load_edf(filename, selected_channels=[]):
    try:

        f = pyedflib.EdfReader(filename)
        # os.remove(filename)

        if len(selected_channels) == 0:
            selected_channels = f.getSignalLabels()

        channel_names = f.getSignalLabels()
        channel_freq = f.getSampleFrequencies()

        sigbufs = np.zeros((f.getNSamples()[0], len(selected_channels)))
        for i, channel in enumerate(selected_channels):
            sigbufs[:, i] = f.readSignal(channel_names.index(channel))

        df = pd.DataFrame(sigbufs, columns=selected_channels).astype('float32')
        index_increase = np.linspace(0, len(df) / channel_freq[0], len(df), endpoint=False)
        seconds = np.floor(index_increase).astype('uint16')
        df['Time'] = seconds
        df = df.set_index('Time')
        df.columns.name = 'Channel'

        return df, channel_freq[0]

    except OSError as e:
        print(f"Error loading EDF file {file}: {e}")
        return pd.DataFrame(), None

        
def chb_mit_parse_summary(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    file_names = []
    start_times = []
    end_times = []
    num_seizures = []
    seizure_start_times = []
    seizure_end_times = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("File Name:"):
            file_name = line.split(":")[1].strip()
            file_names.append(file_name)

            i += 1
            start_time = lines[i].split(":")[1].strip()
            start_times.append(start_time)

            i += 1
            end_time = lines[i].split(":")[1].strip()
            end_times.append(end_time)

            i += 1
            try:
                seizures = int(''.join(filter(str.isdigit, lines[i].split(":")[1].strip())))
            except ValueError:
                seizures = 0  # Default to zero if parsing fails
            num_seizures.append(seizures)


            if seizures > 0:
                i += 1
                start_seizure = int(lines[i].split(":")[1].strip().replace(" seconds", ""))
                seizure_start_times.append(start_seizure)

                i += 1
                end_seizure = int(lines[i].split(":")[1].strip().replace(" seconds", ""))
                seizure_end_times.append(end_seizure)
            else:
                seizure_start_times.append(None)
                seizure_end_times.append(None)
        i += 1

    df = pd.DataFrame({
        'File Name': file_names,
        'Start Time': start_times,
        'End Time': end_times,
        'Number of Seizures': num_seizures,
        'Seizure Start Time': seizure_start_times,
        'Seizure End Time': seizure_end_times
    })
    return df



def chb_mit_add_class_column(summaryRow, csv_path):
    
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_path)

        # Create a mask for times within the seizure window
        time_within_seizure = (
            (data['Time'] >= summaryRow['Seizure Start Time']) & 
            (data['Time'] <= summaryRow['Seizure End Time'])
        )
        
        # Assign the 'Class' column based on the mask
        data['Class'] = time_within_seizure.astype(int)  # 1 for within seizure, 0 otherwise
        data.to_csv(re.sub(".csv", "_classed.csv", csv_path))
        return data

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return pd.DataFrame()  # Return an empty DataFrame if file is not found


def chb_mit_edf_to_csv(data_path):
    for part_file in os.listdir(data_path):
        try:
            file_path = data_path+"/"+part_file      
            df, freq = chb_mit_load_edf(file_path)
            df.to_csv(re.sub(".edf", ".csv", file_path))
            print(f"file processed: {part_file} frequency: {freq}")
        except Exception:
            print (f"error {Exception} ")

def chb_mit_add_class_to_csv(data_dir, summaryDf):
    for index, row in summaryDf.iterrows():
        try:
            file_name = re.sub(".edf", ".csv", row['File Name'])
            csv_path = f"{data_dir}/{file_name}"
            if os.path.exists(csv_path):
                df = chb_mit_add_class_column(row, csv_path)
                print(f"Processed Row {index}: {file_name}" )
        except Exception as e:
            print (f" chb_mit_add_class_to_csv error {str(e)} {file_name}")

# Plot learning curves
def plot_learning_curves(history):
    # Accuracy plot
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_roc_curves(dnn_model, models, X_test, y_test):
    plt.figure(figsize=(10, 8))
    
    # DNN model predictions for ROC
    dnn_pred_prob = dnn_model.predict(X_test).ravel()  # Get probabilities for the DNN model
    fpr, tpr, _ = roc_curve(y_test, dnn_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='DNN (AUC = {:.2f})'.format(roc_auc))

    if models:
        for name, model in models.items():
            y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='{} (AUC = {:.2f})'.format(name, roc_auc))
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()
    
def mat_to_df(file_path, output=False):
    """
    Converts a MATLAB .mat file to a pandas DataFrame.

    Parameters:
        file_path (str): Path to the .mat file.
        output (bool): If True, displays the first few rows of the DataFrame.

    Returns:
        df (pandas.DataFrame): The converted DataFrame.
        freq (float): The frequency value extracted from the .mat file.
    """
    # Load the .mat file
    mat = loadmat(file_path)

    # Extract data, channels, and frequency from the .mat file
    data = mat['data']          # The EEG data
    channels = mat['channels']  # Channel names
    freq = mat['freq'][0]       # Sampling frequency

    # Convert the channels array to a list of strings
    channels_list = [channel[0] for channel in channels[0][0]]

    # Create a DataFrame using the data and channel names as index
    df = pd.DataFrame(data, index=channels_list).T

    # Remove columns where all values are the same (no variation)
    df = df.loc[:, (df != df.iloc[0]).any()]

    # Optionally display the first few rows of the DataFrame
    if output:
        display(df.head())

    return df, freq

# Define color formatting using ANSI escape codes
class color:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


        
def  getPatientUpennFiles(file_path, output=False):
    
    # Use the python os module instead of a shell command to get the file list
    UPENN_P1_DATA_DIR = os.listdir( file_path )
    upenn_P1_file_list = [file.strip() for file in UPENN_P1_DATA_DIR]
    upenn_ictal_list = []
    upenn_interictal_list = []
    for file in upenn_P1_file_list:
      if 'interictal' in file: # Use the 'in' operator to check if substring is present
        upenn_interictal_list.append(file)
      elif 'ictal' in file:
        upenn_ictal_list.append(file)
    
    upenn_seizure_file = upenn_ictal_list[1]
    upenn_baseline_file = upenn_interictal_list[1]
    
    # Define the file paths (replace with your actual paths or variables)
    upenn_seizure_file = file_path + upenn_seizure_file.split('/')[-1]  # Get only the filename
    upenn_baseline_file = file_path + upenn_baseline_file.split('/')[-1] # Get only the filename
    return upenn_baseline_file, upenn_seizure_file


def matToCsv(matPath, csvPath): 
    upenn_baseline_file, upenn_seizure_file = getPatientUpennFiles(matPath)
    
    # Display seizure data with formatting
    print(color.BOLD + color.UNDERLINE + 'Ictal' + color.END)
    upenn_seizure_df, upenn_seizure_freq = mat_to_df(upenn_seizure_file, output=True)
    
    print()
    
    # Display baseline data with formatting
    print(color.BOLD + color.UNDERLINE + 'Interictal' + color.END)
    upenn_baseline_df, upenn_baseline_freq = mat_to_df(upenn_baseline_file, output=True)
    
    # Define the CSV output paths
    seizureAbsPath = csvPath + 'upenn_seizure.csv'
    baselineAbsPath = csvPath + 'upenn_baseline.csv'
    
    # Save DataFrames to CSV
    upenn_seizure_df.to_csv(seizureAbsPath, index=False)
    upenn_baseline_df.to_csv(baselineAbsPath, index=False)
    
    # Return dictionary with frequency and path pairs
    return {
        'seizure': {'frequency': upenn_seizure_freq, 'path': seizureAbsPath},
        'baseline': {'frequency': upenn_baseline_freq, 'path': baselineAbsPath}
    }



# set mne to only output warnings

def mne_object(data, freq):
  # create an mne info file with meta data about the EEG
  info = mne.create_info(ch_names=list(data.columns),
                         sfreq=freq,
                         ch_types=['eeg']*data.shape[-1])

  # data needs to be in volts rather than in microvolts
  data = data.apply(lambda x: x*1e-6)
  # transpose the data
  data_T = data.transpose()

  # create raw mne object
  raw = mne.io.RawArray(data_T, info)

  return raw

def transpose_csv(input_file, output_file):
    """
    Transposes the CSV file: converts columns to rows and rows to columns.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path where the transposed CSV file will be saved.
    """
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Transpose the DataFrame
    transposed_df = df.T

    # Reset the index to make the transposed DataFrame cleaner
    transposed_df.reset_index(inplace=True)

    # Save the transposed DataFrame to a new CSV file
    transposed_df.to_csv(output_file, index=False)


def plotDataFrame(upenn_baseline_df, upenn_seizure_df,upenn_baseline_freq, upenn_seizure_freq):
    mne.set_log_level('WARNING')

    plot_kwargs = {
        'scalings': dict(eeg=20e-5),   # zooms the plot out
        'highpass': 0.5,              # filters out low frequencies
        'lowpass': 70.,                # filters out high frequencies
        'show_scrollbars': False,
        'show': True
    }
    
    print(colored('Interictal', 'white', 'on_grey', attrs=['bold', 'underline'])) # Use colored function from termcolor
    upenn_baseline_mne = mne_object(upenn_baseline_df, upenn_baseline_freq)
    upenn_baseline_mne.plot(**plot_kwargs);
    print()
    print(colored('Ictal', 'white', 'on_grey', attrs=['bold', 'underline'])) # Use colored function from termcolor
    upenn_seizure_mne = mne_object(upenn_seizure_df, upenn_seizure_freq)
    upenn_seizure_mne.plot(**plot_kwargs)\

def plotDataFrameMean(dataFrame):

    # Bar chart
    plt.figure(figsize=(400, 100))
    dataFrame.mean().plot(kind='bar', color='skyblue')
    plt.xlabel("Columns")
    plt.ylabel("Mean Value")
    plt.title("Mean Value of Selected EEG Features")
    plt.xticks(rotation=45)
    plt.show()