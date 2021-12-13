from .object import Object, make_factory
from .proto import api_pb2


class EnvDict(Object):
    def __init__(self, env_dict, session=None):
        super().__init__(session=session)
        self.env_dict = env_dict

    async def _create_impl(self, session):
        req = api_pb2.EnvDictCreateRequest(session_id=session.session_id, env_dict=self.env_dict)
        resp = await session.client.stub.EnvDictCreate(req)
        return resp.env_dict_id


env_dict_factory = make_factory(EnvDict)
