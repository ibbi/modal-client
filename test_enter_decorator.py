import unittest
from modal.partial_function import _PartialFunction, _enter, _PartialFunctionFlags

# Since the decorator is named '_enter' in the import, we need to use that name or alias 'enter' to it.
enter = _enter

class TestEnterDecorator(unittest.TestCase):

    def test_function_without_parentheses(self):
        # Test the behavior of the 'enter' decorator when it is used without parentheses
        print("Running test_function_without_parentheses")
        @enter
        def dummy_function():
            return "Function executed"

        # Assert that the returned object is an instance of _PartialFunction
        self.assertIsInstance(dummy_function, _PartialFunction)
        # Assert that the flag is set correctly
        self.assertEqual(dummy_function.flags, _PartialFunctionFlags.ENTER_POST_CHECKPOINT)

    def test_function_with_parentheses(self):
        # Test the behavior of the 'enter' decorator when it is used with parentheses
        print("Running test_function_with_parentheses")
        @enter()
        def dummy_function():
            return "Function executed"

        # Assert that the returned object is an instance of _PartialFunction
        self.assertIsInstance(dummy_function, _PartialFunction)
        # Assert that the flag is set correctly
        self.assertEqual(dummy_function.flags, _PartialFunctionFlags.ENTER_POST_CHECKPOINT)

if __name__ == "__main__":
    unittest.main()
