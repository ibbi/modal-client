# Copyright Modal Labs 2023
import pytest

from modal import Stub, asgi_app, method, web_endpoint, wsgi_app, enter
from modal.exception import InvalidError
from modal.partial_function import _PartialFunction, _PartialFunctionFlags


def test_local_entrypoint_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="local_entrypoint()"):

        @stub.local_entrypoint  # type: ignore
        def f():
            pass


def test_function_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="function()"):

        @stub.function  # type: ignore
        def f():
            pass


def test_cls_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="cls()"):

        @stub.cls  # type: ignore
        class XYZ:
            pass


def test_method_forgot_parentheses():
    stub = Stub()

    with pytest.raises(InvalidError, match="method()"):

        @stub.cls()
        class XYZ:
            @method  # type: ignore
            def f(self):
                pass


def test_invalid_web_decorator_usage():
    stub = Stub()

    with pytest.raises(InvalidError, match="web_endpoint()"):

        @stub.function()  # type: ignore
        @web_endpoint  # type: ignore
        def my_handle():
            pass

    with pytest.raises(InvalidError, match="asgi_app()"):

        @stub.function()  # type: ignore
        @asgi_app  # type: ignore
        def my_handle_asgi():
            pass

    with pytest.raises(InvalidError, match="wsgi_app()"):

        @stub.function()  # type: ignore
        @wsgi_app  # type: ignore
        def my_handle_wsgi():
            pass


def test_web_endpoint_method():
    stub = Stub()

    with pytest.raises(InvalidError, match="remove the `@method`"):

        @stub.cls()
        class Container:
            @method()  # type: ignore
            @web_endpoint()
            def generate(self):
                pass


def test_enter_decorator_without_parentheses():
    stub = Stub()

    @enter
    def my_function():
        pass

    assert isinstance(my_function, _PartialFunction), "The @enter decorator without parentheses should create an instance of _PartialFunction."
    assert my_function.flag == _PartialFunctionFlags.ENTER_POST_CHECKPOINT, "The @enter decorator without parentheses should default to ENTER_POST_CHECKPOINT flag."

    @enter(snap=True)
    def my_snap_function():
        pass

    assert isinstance(my_snap_function, _PartialFunction), "The @enter decorator with parentheses and snap=True should create an instance of _PartialFunction."
    assert my_snap_function.flag == _PartialFunctionFlags.ENTER_PRE_CHECKPOINT, "The @enter decorator with snap=True should set the ENTER_PRE_CHECKPOINT flag."
