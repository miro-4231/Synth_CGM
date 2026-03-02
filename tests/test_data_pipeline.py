import unittest
from src.data_loader import _generate_sequence, load_dataset

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