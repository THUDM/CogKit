from cogkit.utils import expand_list


class TestExpandList:
    def test_normal_case(self):
        input_data = [{"a": [1], "b": [2]}, {"a": [3], "b": [4]}]
        expected_output = {"a": [1, 3], "b": [2, 4]}
        assert expand_list(input_data) == expected_output

    def test_different_keys(self):
        input_data = [{"a": [1]}, {"b": [2]}, {"c": [3]}]
        expected_output = {"a": [1], "b": [2], "c": [3]}
        assert expand_list(input_data) == expected_output

    def test_empty_list(self):
        input_data = []
        expected_output = {}
        assert expand_list(input_data) == expected_output

    def test_nested_lists(self):
        input_data = [{"a": [[1, 2]], "b": [[3, 4]]}, {"a": [[5, 6]], "b": [[7, 8]]}]
        expected_output = {"a": [[1, 2], [5, 6]], "b": [[3, 4], [7, 8]]}
        assert expand_list(input_data) == expected_output

    def test_mixed_keys(self):
        input_data = [{"a": [1], "b": [2]}, {"a": [3]}, {"b": [4], "c": [5]}]
        expected_output = {"a": [1, 3], "b": [2, 4], "c": [5]}
        assert expand_list(input_data) == expected_output

    def test_empty_values(self):
        input_data = [{"a": [], "b": [1]}, {"a": [2], "b": []}]
        expected_output = {"a": [2], "b": [1]}
        assert expand_list(input_data) == expected_output

    def test_only_list_values(self):
        input_data = [{"a": ["x", "y"]}, {"b": ["z"]}, {"a": ["w"]}]
        expected_output = {"a": ["x", "y", "w"], "b": ["z"]}
        assert expand_list(input_data) == expected_output

    def test_only_single_values(self):
        input_data = [{"a": 1, "b": 2}, {"a": 3, "c": 4}]
        expected_output = {"a": [1, 3], "b": [2], "c": [4]}
        assert expand_list(input_data) == expected_output

    def test_mixed_list_and_single_values(self):
        input_data = [{"a": ["x", "y"], "b": 1}, {"a": 2, "b": ["z", "w"]}]
        expected_output = {"a": ["x", "y", 2], "b": [1, "z", "w"]}
        assert expand_list(input_data) == expected_output
