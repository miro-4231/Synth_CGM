import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from typing import List, Tuple
from typing import Tuple
from scipy import signal 
from datetime import datetime, timedelta
from scipy.integrate import odeint

# Set up logging
logging.basicConfig(level=logging.INFO)

# Choose seed
np.random.seed(42)



def process_segments(timestamps: np.ndarray, values: np.ndarray, delta_time: int, max_timestamps: int) -> List[List[float]]:
    segments = []
    segment = [values[0]]  # Start with first value
    
    # Convert timestamp differences to integers (likely seconds or minutes)
    differences = timestamps[1:].values - timestamps[:-1].values
    # Convert timedelta to the same units as delta_time
    # Assuming delta_time is in seconds, convert nanoseconds to seconds
    differences_int = differences.astype('int64')
    
    for diff, value in zip(differences_int, values[1:]):
        quotient = np.abs(diff / delta_time)
        if quotient < 1.15 :
            # Normal consecutive timestamp
            segment.append(value)
        elif quotient <= max_timestamps:
            # For large gaps, end current segment and start new one
            # instead of filling with None values
            segment.append(value)
        else: 
            # End current segment and start new one
            if segment:
                segments.append(segment)
            segment = [value]            
    
    if segment:
        segments.append(segment)
    return segments


def _generate_sequence(segments: List[list], look_back: int, horizon_prediction: int, classes: List[int], val_size: float, test_size: float, train:bool = True, index:int = None, split:bool = True) -> Tuple:
    sequences = []
    len_seq = look_back + horizon_prediction
    for segment in segments:
        len_seg = len(segment)
        if len_seg > len_seq:
            seq = [segment[i:i + len_seq] for i in range(len_seg - len_seq)]
            sequences.extend(seq)

    sequences = np.array(sequences, dtype=np.float32)
    # split by columns to features and targets (for classification and regression)
    X = sequences[:, :look_back]
    y_reg = sequences[:, look_back:look_back+horizon_prediction]

    num_classes = len(classes) + 1
    y_clf = np.ones_like(y_reg) * int(num_classes)
    for num_class, limit in enumerate(classes):
        y_clf = np.where(y_reg < limit, int(num_classes - 1 - num_class), y_clf)

    if index is None: 
        y_clf = y_clf.min(axis = 1) 
    elif isinstance(index, list):
        y_clf = y_clf[:, index].min(axis = 1) 
    else:
        assert index < horizon_prediction
        y_clf = y_clf[:, index]
    y_clf = y_clf - 1
    
    if not train:
        return X, y_reg, y_clf
    
    # train val test split
    if split:
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(X, y_reg, y_clf, test_size=val_size, shuffle=False)

        if test_size is not None:
            X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val = train_test_split(X_train, y_reg_train, y_clf_train, test_size=test_size, shuffle=False)

            return X_train, X_val, X_test, y_reg_train, y_reg_val, y_reg_test, y_clf_train, y_clf_val, y_clf_test
        else:
            return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test
    else:
        return X, y_reg, y_clf


def load_dataset(config: dict, base_path_data: str, dataset_name: str, look_back: int = 30, horizon_prediction: int = 6, max_timestamps: int = 2, classes: list = [180, 60], val_size: float = 0.2, test_size: float = 0.3, return_segments:bool = False, index:int = None) -> Tuple:
    """
    Get organized time series data for training, evaluation and testing
    """
    # Check if the file exists
    dataset_path = os.path.join(base_path_data, config[dataset_name]['path'])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Read csv file
    dataset = pd.read_csv(dataset_path, sep=config[dataset_name]['sep'])

    # Parse time column
    dataset[config[dataset_name]["time_column"]] = pd.to_datetime(dataset[config[dataset_name]["time_column"]], format=config[dataset_name]["date_time_format"])

    # Divide the dataset based on patientId
    DFs = []
    for patientID in dataset['PtID'].unique():
        DFs.append(dataset[dataset['PtID'] == patientID].sort_values(by=config[dataset_name]["time_column"], ascending=True))

    # Process segments
    delta_time = np.int64(5) * np.int64(60000000000)
    segments = []
    segment = []

    for df in DFs:
        timestamps = df[config[dataset_name]["time_column"]]
        values = df[config[dataset_name]["target_column"]].values
        segments.extend(process_segments(timestamps, values, delta_time, max_timestamps))

    if return_segments:
        return segments

    return _generate_sequence(segments, look_back, horizon_prediction, classes, val_size, test_size, index = index)


def load_OhioT1DM_patient_split(path: str, look_back: int = 30, horizon_prediction: int = 6, max_timestamps: int = 2, train: bool = True, classes: list = [180, 60], val_size: float = 0.1, test_size: float = 0.3, return_segments:bool = False, index:int = None, split:bool = True) -> Tuple:
    """
    Load OhioT1DM dataset (from XML files) and preprocess it into sequences
    """
    # Retrieve XML file paths
    xml_file_paths = glob.glob(os.path.join(path, "OhioT1DM", "**", "**.xml"), recursive=True)
    data = {"train": [], "test": []}
    root_path = path
    # Parse XML files
    for path in xml_file_paths:
        sequence = {}
        split = path.split("\\")[1 + len(root_path.split("\\"))]
        tree = ET.parse(path)
        root = tree.getroot()

        # Extract patient attributes
        patient_id = root.attrib.get('id')
        weight = root.attrib.get('weight')
        insulin_type = root.attrib.get('insulin_type')

        # Write glucose events
        for event in root.find('glucose_level'):
            timestamp = event.attrib.get('ts')
            value = event.attrib.get('value')
            sequence[timestamp] = value

        data[split].append({"patient_id": patient_id, "weight": weight, "insulin_type": insulin_type, "sequence": sequence})

    # Create DataFrames
    DF = []
    for _, split_data in data.items():
        for patient_data in split_data:
            patient_id = patient_data["patient_id"]
            weight = patient_data["weight"]
            insulin_type = patient_data["insulin_type"]
            sequence = patient_data["sequence"]
            timestamps = list(sequence.keys())
            values = [float(value) for value in sequence.values()]
            num_rows = len(values)
            df = {
                "id": [np.int32(patient_id)] * num_rows,
                "weight": [np.int32(weight)] * num_rows,
                "insulin_type": [insulin_type] * num_rows,
                "timestamp": timestamps,
                "value": np.float32(values)
            }
            df = pd.DataFrame(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
            DF.append(df)


    delta_time = np.int64(5) * np.int64(60000000000)
    segments = []
    sequences  = []

    for df in DF:
        timestamps = df["timestamp"]# .values
        values = df["value"].values
        segments.append(process_segments(timestamps, values, delta_time, max_timestamps))

    if return_segments:
        return segments
    
    for patient_segments in segments: 
        patient_sequence = _generate_sequence(patient_segments, look_back, horizon_prediction, classes, val_size, test_size, index=index, train=train, split=split) 
        sequences.append(patient_sequence)
        
    final_data = []
        
    n = len(patient_sequence) 
    
    for i in range(n): 
        temp = []
        for patient_sequence in sequences: 
            temp.append(patient_sequence[i])
            
        final_data.append(np.concatenate(temp))
    
    return final_data


def load_OhioT1DM(path: str, look_back: int = 30, horizon_prediction: int = 6, max_timestamps: int = 2, train: bool = True, classes: list = [180, 60], val_size: float = 0.2, test_size: float = 0.3, return_segments:bool = False, index:int = None, split:bool = True) -> Tuple:
    """
    Load OhioT1DM dataset (from XML files) and preprocess it into sequences
    """
    # Retrieve XML file paths
    xml_file_paths = glob.glob(os.path.join(path, "OhioT1DM", "**", "**.xml"), recursive=True)
    data = {"train": [], "test": []}
    root_path = path
    # Parse XML files
    for path in xml_file_paths:
        sequence = {}
        split = path.split("\\")[1 + len(root_path.split("\\"))]
        tree = ET.parse(path)
        root = tree.getroot()

        # Extract patient attributes
        patient_id = root.attrib.get('id')
        weight = root.attrib.get('weight')
        insulin_type = root.attrib.get('insulin_type')

        # Write glucose events
        for event in root.find('glucose_level'):
            timestamp = event.attrib.get('ts')
            value = event.attrib.get('value')
            sequence[timestamp] = value

        data[split].append({"patient_id": patient_id, "weight": weight, "insulin_type": insulin_type, "sequence": sequence})

    # Create DataFrames
    DFs = {}
    for split, split_data in data.items():
        DF = []
        for patient_data in split_data:
            patient_id = patient_data["patient_id"]
            weight = patient_data["weight"]
            insulin_type = patient_data["insulin_type"]
            sequence = patient_data["sequence"]
            timestamps = list(sequence.keys())
            values = [float(value) for value in sequence.values()]
            num_rows = len(values)
            df = {
                "id": [np.int32(patient_id)] * num_rows,
                "weight": [np.int32(weight)] * num_rows,
                "insulin_type": [insulin_type] * num_rows,
                "timestamp": timestamps,
                "value": np.float32(values)
            }
            df = pd.DataFrame(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M:%S")
            DF.append(df)
        DFs[split] = DF

    delta_time = np.int64(5) * np.int64(60000000000)
    segments = []
    segment = []

    for df in DFs["train" if train else "test"]:
        timestamps = df["timestamp"]# .values
        values = df["value"].values
        segments.extend(process_segments(timestamps, values, delta_time, max_timestamps))

    if return_segments:
        return segments
    
    return _generate_sequence(segments, look_back, horizon_prediction, classes, val_size, test_size, index=index, train=train, split=split)

def parse_timestamp(ts_str):
    return datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")

def insulin_model_ode(y, t, k_a, k_e, bolus_times, bolus_doses):
    I_SC, I_plasma = y
    bolus_input = sum(
        dose for bt, dose in zip(bolus_times, bolus_doses) if abs(t - bt) < 1e-3
    )
    dI_SC_dt = -k_a * I_SC + bolus_input
    dI_plasma_dt = k_a * I_SC - k_e * I_plasma
    return [dI_SC_dt, dI_plasma_dt]

def iap_approximation(t_diff_minutes, onset=0, peak=60, duration=240):
    if t_diff_minutes < onset or t_diff_minutes > duration:
        return 0
    elif t_diff_minutes <= peak:
        return (t_diff_minutes - onset) / (peak - onset)
    else:
        return (duration - t_diff_minutes) / (duration - peak)

def parse_dataset(xml_path, insulin_model='none', decay_rate=0.9, k_a=1.0, k_e=0.1, meal_window_minutes=5):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Parse glucose readings
    glucose_data = []
    glucose_tag = root.find("glucose_level")
    if glucose_tag is not None:
        for event in glucose_tag.findall("event"):
            ts = parse_timestamp(event.attrib["ts"])
            value = float(event.attrib["value"])
            glucose_data.append({"timestamp": ts, "glucose": value})
    glucose_df = pd.DataFrame(glucose_data).sort_values("timestamp").reset_index(drop=True)

    # Parse bolus insulin events
    bolus_data = []
    bolus_tag = root.find("bolus")
    if bolus_tag is not None:
        for event in bolus_tag.findall("event"):
            ts = parse_timestamp(event.attrib["ts_begin"])
            dose = float(event.attrib.get("dose", 0))
            bolus_data.append({"timestamp": ts, "insulin": dose})
    bolus_df = pd.DataFrame(bolus_data)

    # Parse meal events
    meal_data = []
    meal_tag = root.find("meal")
    if meal_tag is not None:
        for event in meal_tag.findall("event"):
            ts = parse_timestamp(event.attrib["ts"])
            carbs = float(event.attrib.get("carbs", 0))
            meal_data.append({"timestamp": ts, "carbs": carbs})
    meal_df = pd.DataFrame(meal_data)

    result = []

    for i, row in glucose_df.iterrows():
        t = row["timestamp"]

        # Filter bolus events before current glucose reading
        insulin_events = bolus_df[bolus_df["timestamp"] <= t]

        # Filter meal events within the specified time window before the current glucose reading
        if not meal_df.empty:
            meal_events = meal_df[
                (meal_df["timestamp"] <= t) &
                (meal_df["timestamp"] >= t - timedelta(minutes=meal_window_minutes))
            ]
        else: 
            meal_events = pd.DataFrame({'carbs':[0]})

        # Calculate insulin_before based on selected model
        if insulin_model == 'none':
            insulin_total = insulin_events["insulin"].sum()
        elif insulin_model == 'decay':
            insulin_total = sum(
                r["insulin"] * (decay_rate ** ((t - r["timestamp"]).total_seconds() / 3600))
                for _, r in insulin_events.iterrows()
            )
        elif insulin_model == 'two_compartment':
            if insulin_events.empty:
                insulin_total = 0
            else:
                start_time = insulin_events["timestamp"].min()
                end_time = t
                total_minutes = int((end_time - start_time).total_seconds() / 60)
                time_points = [start_time + timedelta(minutes=i) for i in range(total_minutes + 1)]
                time_hours = np.array([(tp - start_time).total_seconds() / 3600 for tp in time_points])

                bolus_times = [(bt - start_time).total_seconds() / 3600 for bt in insulin_events["timestamp"]]
                bolus_doses = insulin_events["insulin"].tolist()

                y0 = [0, 0]  # I_SC, I_plasma

                solution = odeint(insulin_model_ode, y0, time_hours, args=(k_a, k_e, bolus_times, bolus_doses))
                I_plasma_series = solution[:, 1]

                insulin_total = I_plasma_series[-1]
        elif insulin_model == 'iap':
            insulin_total = 0
            for _, r in insulin_events.iterrows():
                t_diff = (t - r["timestamp"]).total_seconds() / 60  # in minutes
                activity = iap_approximation(t_diff)
                insulin_total += r["insulin"] * activity
        else:
            raise ValueError("Invalid insulin_model. Choose from 'none', 'decay', 'two_compartment', or 'iap'.")
        if meal_events.empty:
            carbs_total = 0 
        else:
            carbs_total = meal_events["carbs"].sum()

        result.append({
            "timestamp": t,
            "glucose": row["glucose"],
            "carbs_before": carbs_total,
            "insulin_before": insulin_total
        })

    return pd.DataFrame(result)


def load_ohio_T1DM_insulin_cho(path: str, data_parse_args: dict = { "insulin_model": "iap" }, look_back: int = 30, horizon_prediction: int = 6, max_timestamps: int = 2,
                               train: bool = True, classes: list = [180, 60], val_size: float = 0.2, test_size: float = 0.3,
                               return_segments:bool = False, index:int = None, split:bool = True) -> Tuple:
    xml_file_paths = glob.glob(os.path.join(path, "OhioT1DM", "**", "**.xml"), recursive=True)
    data = {"train": [], "test": []}

    # Parse XML files
    for path in xml_file_paths:
        # print(path)
        sequence = {}
        split = path.split("\\")[3]
        tree = ET.parse(path)
        root = tree.getroot()

        # Extract patient attributes
        patient_id = root.attrib.get('id')
        weight = root.attrib.get('weight')
        insulin_type = root.attrib.get('insulin_type')
        data_parse_args ["xml_path"] = path
        sequence_data = parse_dataset(**data_parse_args)

        data[split].append({"patient_id": patient_id, "weight": weight, "insulin_type": insulin_type, "sequence": sequence_data})   
    
    
    delta_time = np.int64(5) * np.int64(60000000000)
    segments = []

    for df in data["train" if train else "test"]:
        df = df["sequence"]
        timestamps = df["timestamp"]
        values = df[["glucose",  "carbs_before" , "insulin_before"]].values
        segments.extend(process_segments(timestamps, values, delta_time, max_timestamps))

    if return_segments: 
        return segments
    
    return _generate_sequence(segments, look_back, horizon_prediction, classes, val_size, test_size, index = index, train=train)



def load_generated(file_path, LBW=10, PH=6, classes:list=[180, 60], index:int=None):
        generated_sequences = np.load(file_path)
        assert LBW + PH <= generated_sequences.shape[1]
        # split by columns to features and targets (for classification and regression)
        X = generated_sequences[:, :LBW]
        y_reg = generated_sequences[:, LBW:LBW+PH]
        num_classes = len(classes) + 1
        y_clf = np.ones_like(y_reg) * int(num_classes)
        for num_class, limit in enumerate(classes):
                y_clf = np.where(y_reg < limit, int(num_classes - 1 - num_class), y_clf)

        if index is None: 
                y_clf = y_clf.min(axis = 1) 
        elif isinstance(index, list):
                y_clf = y_clf[:, index].min(axis = 1) 
        else:
                assert index < PH
                y_clf = y_clf[:, index]
        y_clf = y_clf - 1
        
        # train val test split

        return train_test_split(X, y_reg, y_clf, test_size=0.2, shuffle=False)

def add_delta_measure(X:np.ndarray) -> np.ndarray:
    assert len(X.shape) == 2
    return np.stack([X[:, 1:], np.diff(X, axis = 1)], axis = 2)

def _resample(data:np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    return signal.resample(data, data.shape[1] * 5, axis=1)


def _get_delay(y:np.ndarray, limit:float) -> np.ndarray:
    zero_crossings = np.where(np.diff(np.int32(np.signbit(y - limit)), axis = 1)==-1)
    df = pd.DataFrame(zero_crossings).T
    out = df.groupby(0).min().reset_index().values
    return out

def get_delay_dataset(X:np.ndarray, y:np.ndarray, limit:float = 70, resample:bool = True) -> Tuple[np.ndarray, np.ndarray] : 
    
    if resample:
        X = _resample(X)
        y = _resample(y)
    
    delay = _get_delay(y, limit)
    
    X = X[delay[:, 0]]
    y = delay[:, 1]
    
    return X, y



# Unit tests example
import unittest

class TestDatasetFunctions(unittest.TestCase):

    def test_generate_sequence(self):
        segments = [[1, 2, 3, 4, 5, 6, 4, 5, 2, 7], [7, 8, 9, 10, 11, 12, 2, 54,  1223,  2,3, 123]] * 100
        look_back = 4
        horizon_prediction = 2
        classes = [6, 8]
        val_size = 0.2
        test_size = 0.3
        X_train, X_val, X_test, y_reg_train, y_reg_val, y_reg_test, y_clf_train, y_clf_val, y_clf_test = _generate_sequence(segments, look_back, horizon_prediction, classes, val_size, test_size)
        
        # Test shapes
        self.assertEqual(len(X_train.shape), 2)  # Example expected value
        self.assertEqual(y_reg_train.shape[1], horizon_prediction)

    def test_load_dataset(self):
        
        International_path = "cgm.txt"
        IOBP_path = "IOBP2DeviceCGM.txt"
        config = {
    'International': {
        'path': International_path,
        'sep': '|',
        'date_time_format': "%d%b%y:%H:%M:%S",
        'time_column': 'DataDtTm',
        'target_column': 'CGM',
    },
    'IOBP': {
        'path': IOBP_path,
        'sep': '|',
        'date_time_format': None,
        'time_column': 'DeviceDtTm',
        'target_column': 'Value',
    },
        }
        base_path_data = 'D:\Projects\PythonProjects\PFE_CAD\CGM_PROJECT\Data'
        dataset_name = 'International'
        look_back = 30
        horizon_prediction = 6
        segments = load_dataset(config, base_path_data, dataset_name, look_back, horizon_prediction)
        self.assertIsInstance(segments, tuple)


if __name__ == "__main__":
    unittest.main()
