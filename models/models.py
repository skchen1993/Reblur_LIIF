import copy


models = {}
# dict, 裝著各個model對應的class, 等等用於建立對應的object

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        # model_args => 是從剛剛的 config['models'] 來, 裝了 LIIF (其下的args: EDSR + MLP) 的設定訊息
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args) # 同dataset 那裡的建立方法, 'name'=='liif', 傳入的是 EDSR, imnet(MLP) 資料
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
