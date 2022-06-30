import pytest

import modal.exception
from modal import Stub
from modal.aio import AioDebianSlim, AioQueue, AioRunningApp, AioStub, aio_lookup
from modal.exception import NotFoundError, VersionError


@pytest.mark.asyncio
async def test_kwargs(servicer, aio_client):
    stub = AioStub(
        q1=AioQueue(),
        q2=AioQueue(),
    )
    async with stub.run(client=aio_client) as app:
        await app["q1"].put("foo")
        await app["q2"].put("bar")
        assert await app["q1"].get() == "foo"
        assert await app["q2"].get() == "bar"


@pytest.mark.asyncio
async def test_attrs(servicer, aio_client):
    stub = AioStub()
    stub.q1 = AioQueue()
    stub.q2 = AioQueue()
    async with stub.run(client=aio_client) as app:
        await app.q1.put("foo")
        await app.q2.put("bar")
        assert await app.q1.get() == "foo"
        assert await app.q2.get() == "bar"


@pytest.mark.asyncio
async def test_create_object(servicer, aio_client):
    stub = AioStub()
    async with stub.run(client=aio_client) as running_app:
        q = await AioQueue().create(running_app)
        await q.put("foo")
        await q.put("bar")
        assert await q.get() == "foo"
        assert await q.get() == "bar"


@pytest.mark.asyncio
async def test_persistent_object(servicer, aio_client):
    stub_1 = AioStub()
    stub_1["q_1"] = AioQueue()
    await stub_1.deploy("my-queue", client=aio_client)

    stub_2 = AioStub()
    async with stub_2.run(client=aio_client) as running_app_2:
        assert isinstance(running_app_2, AioRunningApp)

        # Supported in (modal<=0.0.18), remove after bumping version
        with pytest.raises(VersionError):
            await running_app_2.include("my-queue")

        q_3 = await aio_lookup("my-queue", client=aio_client)
        assert isinstance(q_3, AioQueue)
        assert q_3.object_id == "qu-1"

        with pytest.raises(NotFoundError):
            await aio_lookup("bazbazbaz", client=aio_client)


def square(x):
    return x**2


@pytest.mark.asyncio
async def test_redeploy(servicer, aio_client):
    stub = AioStub()
    stub.function(square)
    f_name = "client_test.stub_test.square"

    # Deploy app
    app_id = await stub.deploy("my-app", client=aio_client)
    assert app_id == "ap-1"
    assert servicer.app_objects["ap-1"][f_name] == "fu-1"

    # Redeploy, make sure all ids are the same
    app_id = await stub.deploy("my-app", client=aio_client)
    assert app_id == "ap-1"
    assert servicer.app_objects["ap-1"][f_name] == "fu-1"

    # Deploy to a different name, ids should change
    app_id = await stub.deploy("my-app-xyz", client=aio_client)
    assert app_id == "ap-2"
    assert servicer.app_objects["ap-2"][f_name] == "fu-2"


# Should exit without waiting for the logs grace period.
@pytest.mark.timeout(1)
def test_create_object_exception(servicer, client):
    servicer.function_create_error = True

    stub = Stub()

    @stub.function
    def f():
        pass

    with pytest.raises(Exception):
        with stub.run(client=client):
            pass


def test_deploy_falls_back_to_app_name(servicer, client):
    named_stub = Stub(name="foo_app")
    named_stub.deploy(client=client)
    assert "foo_app" in servicer.deployed_apps


def test_deploy_uses_deployment_name_if_specified(servicer, client):
    named_stub = Stub(name="foo_app")
    named_stub.deploy("bar_app", client=client)
    assert "bar_app" in servicer.deployed_apps
    assert "foo_app" not in servicer.deployed_apps


@pytest.mark.skip(reason="revisit in a sec once the app state stuff is fixed")
def test_run_function_without_app_error():
    stub = Stub()

    @stub.function()
    def foo():
        pass

    with pytest.raises(modal.exception.InvalidError):
        foo()


@pytest.mark.asyncio
async def test_standalone_object(aio_client):
    stub = AioStub()
    image = AioDebianSlim()

    @stub.function
    def foo(image=image):
        pass

    async with stub.run(client=aio_client):
        pass


@pytest.mark.asyncio
async def test_is_inside_basic():
    stub = AioStub()
    assert stub.is_inside() is False
