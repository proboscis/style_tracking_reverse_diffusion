import uuid

from pinject_design import Injected
from pinject_design.di.graph import IObjectGraph
from strd.ldm_img2img import get_or_create_txt2img_env_v2


@Injected.bind
def txt2img_env(session: IObjectGraph):
    factory = session.factory

    @Injected.bind
    def resource(gpu):
        uid = str(uuid.uuid4())[:5]
        return factory.create()[get_or_create_txt2img_env_v2(name=f"txt2img_env_{uid}")]

    return resource
