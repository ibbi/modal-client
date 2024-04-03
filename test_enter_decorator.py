import unittest
from unittest.mock import patch
from modal import enter

class TestEnterDecorator(unittest.TestCase):

    def test_function_without_parentheses(self):
        with patch('modal._PartialFunction') as mock_partial:
            @enter
            def dummy_function():
                return "Function executed"

            dummy_function()
            mock_partial.assert_called_once()

    def test_function_with_parentheses(self):
        with patch('modal._PartialFunction') as mock_partial:
            @enter()
            def dummy_function():
                return "Function executed"

            dummy_function()
            mock_partial.assert_called_once()

if __name__ == "__main__":
    unittest.main()
