from syndi.data import load_demo_data

@pytest.mark.usefixtures("change_test_dir")
class TestData(unittest.TestCase):
    def test_load_data(self):
        load_demo_data()