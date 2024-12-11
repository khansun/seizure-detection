def get_content(part_code):
  url = "https://physionet.org/physiobank/database/chbmit/"+part_code+'/'+part_code+'-summary.txt'
  filename = "./chbmit.txt"

  urlretrieve(url,filename)

  # read the file into a list
  with open(filename, encoding='UTF-8') as f:
      # read all the document into a list of strings (each line a new string)
      content = f.readlines()
      os.remove(filename)

  return content

  import re

def info_dict(content):
  
  line_nos=len(content)
  line_no=1

  channels = []
  file_name = []
  file_info_dict={}

  for line in content:

    # if there is Channel in the line...
    if re.findall('Channel \d+', line):
      # split the line into channel number and channel reference
      channel = line.split(': ')
      # get the channel reference and remove any new lines
      channel = channel[-1].replace("\n", "")
      # put into the channel list
      channels.append(channel)

    # if the line is the file name
    elif re.findall('File Name', line):
      # if there is already a file_name
      if file_name:
        # flush the current file info to it
        part_info_dict[file_name] = file_info_dict

      # get the file name
      file_name = re.findall('\w+\d+_\d+|\w+\d+\w+_\d+', line)[0]

      file_info_dict = {}
      # put the channel list in the file info dict and remove duplicates
      file_info_dict['Channels'] = list(set(channels))
      # reset the rest of the options
      file_info_dict['Start Time'] = ''
      file_info_dict['End Time'] = ''
      file_info_dict['Seizures Window'] = []

    # if the line is about the file start time
    elif re.findall('File Start Time', line):
      # get the start time
      file_info_dict['Start Time'] = re.findall('\d+:\d+:\d+', line)[0]

    # if the line is about the file end time
    elif re.findall('File End Time', line):
      # get the start time
      file_info_dict['End Time'] = re.findall('\d+:\d+:\d+', line)[0]

    elif re.findall('Seizure Start Time|Seizure End Time|Seizure \d+ Start Time|Seizure \d+ End Time', line):
      file_info_dict['Seizures Window'].append(int(re.findall('\d+', line)[-1]))

    # if last line in the list...
    if line_no == line_nos:
      # flush the file info to it
      part_info_dict[file_name] = file_info_dict

    line_no+=1
    

def data_load(file, selected_channels=[]):

  try: 
    url = "https://physionet.org/physiobank/database/chbmit/"+file
    filename = "./chbmit.edf"

    urlretrieve(url,filename)
    # use the reader to get an EdfReader file
    f = pyedflib.EdfReader(filename)
    os.remove(filename)
    
    # get a list of the EEG channels
    if len(selected_channels) == 0:
      selected_channels = f.getSignalLabels()

    # get the names of the signals
    channel_names = f.getSignalLabels()
    # get the sampling frequencies of each signal
    channel_freq = f.getSampleFrequencies()

    # make an empty file of 0's
    sigbufs = np.zeros((f.getNSamples()[0],len(selected_channels)))
    # for each of the channels in the selected channels
    for i, channel in enumerate(selected_channels):
      # add the channel data into the array
      sigbufs[:, i] = f.readSignal(channel_names.index(channel))
    
    # turn to a pandas df and save a little space
    df = pd.DataFrame(sigbufs, columns = selected_channels).astype('float32')
    
    # get equally increasing numbers upto the length of the data depending
    # on the length of the data divided by the sampling frequency
    index_increase = np.linspace(0,
                                  len(df)/channel_freq[0],
                                  len(df), endpoint=False)

    # round these to the lowest nearest decimal to get the seconds
    seconds = np.floor(index_increase).astype('uint16')

    # make a column the timestamp
    df['Time'] = seconds

    # make the time stamp the index
    df = df.set_index('Time')

    # name the columns as channel
    df.columns.name = 'Channel'

    return df, channel_freq[0]

  except:
    OSError
    return pd.DataFrame(), None


def create_events(file_name, df):

  # default df with all 0's the length of the last time datapoint
  events_df = pd.DataFrame([0]*int(df.iloc[-1].name+1))

  if file_name in part_info_dict:
    seizures_window = part_info_dict[file_name]['Seizures Window']
  else:
    seizures_window = []

  # if not empty
  if seizures_window:
    # just get the start values
    seizures_start = seizures_window[0::2]
    # just get the end values
    seizures_end = seizures_window[1::2]
    # get index values where seizures occour
    seizures_index = []
    for index, value in enumerate(seizures_start):
      seizures_index.extend(list(range(value,seizures_end[index])))

    # make a df of 1's with the index where there are seizures
    seizure_events_df = pd.DataFrame([1]*len(seizures_index),seizures_index)

    # update values where seizure
    events_df.update(seizure_events_df, overwrite = True)

  # name the index 'time'
  events_df.index.name='Time'
  # name the column event
  events_df.columns=['Class']

  events_df = events_df.astype('uint8')
  
  # join the events to match the time they are associated with
  df = df.join(events_df, on='Time')

  # get the events
  data_y = df['Class']

  return data_y

def window_y(events, window_size, overlap):
    
  # window the data so each row is another epoch
  events_windowed = window(events, w = window_size, o = overlap, copy = True)

  # get the value most frequent in the window
  #data_y_mode = stats.mode(events_windowed, axis=1)[0].flatten()
  
  # turn to array of bools if seizure in the
  # windowed data
  bools = events_windowed == 1
  # are there any seizure seconds in the data?
  data_y = np.any(bools,axis=1)
  # turn to 0's and 1's
  data_y = data_y.astype(int)

  
  return data_y

def save_to_database(save_dir, file_title, group, data_x, data_y, feature_columns):

    # open the file in append mode (make it if doesnt exist)
    h5file = tables.open_file(save_dir, mode="a", title=file_title)
    
    # save space
    data_x = data_x.astype(np.float32)
    data_y = data_y.astype(np.int16)
    
    # get filters to compress file
    filters = tables.Filters(complevel=1, complib='zlib')
    
    # if there is already a node for the particpant...
    if "/"+group in h5file:
        # ...put in the directory of where it is found
        part_x_array = h5file.get_node("/" + group + '/Data_x')
        part_y_array = h5file.get_node("/" + group + '/Data_y')
    
    else:
        # create the group directory
        part_group = h5file.create_group("/", group, 'Group Data')
        # make an atom which has the datatype found in the data we want to store
        x_atom = tables.Atom.from_dtype(data_x.dtype)
        y_atom = tables.Atom.from_dtype(data_y.dtype)

        # create an array we can append onto later
        part_x_array = h5file.create_earray("/" + group,                   # parentnode
                                            'Data_x',                        # name 
                                            x_atom,                          # atom
                                            (0,data_x.shape[1], data_x.shape[2]), # shape
                                            'Feature Array',
                                            filters=filters
                                           )                 # title

        part_y_array = h5file.create_earray("/" + group, 
                                            'Data_y', 
                                            y_atom, 
                                            (0,),
                                            'Events Array',
                                            filters=filters
                                           )
        
        # create the feature names array (we only need to do this once)
        h5file.create_array("/" + group,                                   # where
                            'Feature_Names',                             # name 
                            np.array(feature_columns, dtype='unicode'),  # obj
                            "Names of Each Feature")                         # title
    
    # append the data to the array directory
    part_x_array.append(data_x)
    part_y_array.append(data_y)
    
    # flush the data to disk
    h5file.flush()
    # close the file
    h5file.close()


if CHB_FILT_OVERWRITE or CHB_FEAT_OVERWRITE:

  if CHB_FILT_OVERWRITE and os.path.exists(CHB_FILT_SAVE_PATH):
    os.remove(CHB_FILT_SAVE_PATH)

  if CHB_FEAT_OVERWRITE and os.path.exists(CHB_FEAT_SAVE_PATH):
    os.remove(CHB_FEAT_SAVE_PATH)

  dbs = wfdb.get_dbs()

  records_list = wfdb.io.get_record_list('chbmit', records='all')
  part_codes = sorted(list(set([record.split('/')[0] for record in records_list])))

  # these are channels common to all records
  channel_keeps = ['P7-O1','FP2-F4','F7-T7','FP1-F3','F4-C4','FP2-F8','P8-O2',
                  'C4-P4','F3-C3','T8-P8','P3-O1','FZ-CZ','FP1-F7','P4-O2',
                  'T7-P7','CZ-PZ','F8-T8','C3-P3']
  
  # ----------------------------
  # PARTICIPANT INFORMATION DICT
  # ----------------------------
  part_info_dict = {}
  for part_code in part_codes:
    content = get_content(part_code)
    info_dict(content)

  # fix wrong times
  start_time_24 = part_info_dict['chb03_24']['Start Time']
  end_time_24 = part_info_dict['chb03_24']['End Time']
  start_time_25 = part_info_dict['chb03_25']['Start Time']
  end_time_25 = part_info_dict['chb03_25']['End Time']
  part_info_dict['chb03_24']['Start Time'] = start_time_25
  part_info_dict['chb03_24']['End Time'] = end_time_25
  part_info_dict['chb03_25']['Start Time'] = start_time_24
  part_info_dict['chb03_25']['End Time'] = end_time_24

  # ---------------
  # CREATE FEATURES
  # ---------------
  # reduce the list to just the first participant
  if not CHB_ALL_PART:
    regex = re.compile('chb01')
    records_list = [i for i in records_list if regex.search(i)]

  for record in tqdm(records_list):
    part_id = record.split('/')[0]
    file_name = record.split('/')[1].split('.')[0]
    raw_data, freq = data_load(record, channel_keeps)
    if raw_data.empty:
      print('Skipped: '+file_name)
    else:
      raw_events = create_events(file_name, raw_data)
      
      if CHB_FILT_OVERWRITE:
        # filter the data
        b, a = signal.butter(4, [1/(freq/2), 30/(freq/2)], 'bandpass', analog=False)
        filt_x = signal.filtfilt(b, a, raw_data.T).T
        
        # scale the data
        SS = StandardScaler()
        scaled_data = SS.fit_transform(filt_x)
        scaled_data = pd.DataFrame(scaled_data, columns = raw_data.columns, 
                                   index = raw_events)
        # drop na
        scaled_data = scaled_data.dropna()
        
        # window the data
        window_size = 256*2
        overlap = 256
        data_x = window_x(scaled_data, window_size, overlap)
        data_y = window_y(raw_events, window_size, overlap)
        
        # save the data
        save_to_database(CHB_FILT_SAVE_PATH,
                        'CHB_filt',
                        part_id,
                        data_x,
                        data_y, 
                        list(scaled_data.columns))
      
      if CHB_FEAT_OVERWRITE:
        feat = Seizure_Features(sf = freq,
                                window_size=2,
                                overlap=1,
                                levels=6,
                                bandpasses = [[1,4],[4,8],[8,12],
                                              [12,30],[30,70]],
                                feature_list=['power', 'power_ratio', 
                                              'mean', 
                                              'mean_abs', 'std', 'ratio', 
                                              'LSWT', 
                                              'fft_corr', 'fft_eigen', 'time_corr', 
                                              'time_eigen'],
                                scale = True)
        with warnings.catch_warnings():
          warnings.filterwarnings("ignore", category=RuntimeWarning)
          x_feat, y_feat = feat.transform(raw_data.values, 
                                          raw_events.values,
                                          channel_names_list = list(raw_data.columns))

        
        # TODO: sometimes the feats come out as Float64Index...
        # should probably look into why that is, but for now, I'll just
        # turn them to ints
        feat_df = pd.DataFrame(x_feat,
                               index=y_feat[:,0].astype(int), 
                               columns = feat.feature_names)

        feat_df = feat_df.dropna()
        
        feat_df.to_hdf(CHB_FEAT_SAVE_PATH, part_id, format='table', append=True)