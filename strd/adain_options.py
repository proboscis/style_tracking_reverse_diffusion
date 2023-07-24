from pinject_design import Injected
from strd.conv_util import list_convs


def get_adain_options(sch):
    with sch["img2img_env"] as img2img_env:
        tgt = img2img_env["img2img_vars"]["model"].model
        convs = img2img_env.put(list_convs)(tgt).fetch()
        skips = [c for c in convs if "skip" in c]
        input_layers = [c for c in convs if "input" in c]
        output_layers = [c for c in convs if "output" in c]
        middles = [c for c in convs if "middle" in c]
        proj_ins = [c for c in convs if "proj_in" in c]
        proj_outs = [c for c in convs if "proj_out" in c]
        in_layers = [c for c in convs if "in_layers" in c]
        out_layers = [c for c in convs if "out_layers" in c]
        ops = [c for c in convs if "op" in c]
        conv_layers = [c for c in convs if "conv_layers" in c]
        adain_options = dict(
            skips=skips,
            input_layers=input_layers,
            output_layers=output_layers,
            middles=middles,
            proj_ins=proj_ins,
            proj_outs=proj_outs,
            in_layers=in_layers,
            out_layers=out_layers,
            ops=ops,
            conv_layers=conv_layers,
            all=convs
        )
        return adain_options

adain_options = Injected.bind(get_adain_options)
