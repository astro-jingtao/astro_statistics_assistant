import pytest
from asa.utils import list_reshape


class TestUtils:
    def test_list_reshape(self):
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        shape = (3, 3)
        expected_output = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert list_reshape(lst, shape) == expected_output


        lst = [1, 2, 3, 4, 5, 6, 7, 8]
        shape = (2, 4)
        expected_output = [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert list_reshape(lst, shape) == expected_output