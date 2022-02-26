import copy


datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(dataset_spec, args=None):
    # 第一次的make中，dataset_spec 是 train_dataset 下 dataset 這個dict, 裡面有name, args
    # 第二次的make中，dataset_spec 是 train_dataset 下 wrapper 這個dict, 裡面有name, args, 並且 args 這個dict裝的是第一次make做出來的 class ImageFolder object
    if args is not None:
        # copy module 裡面有分shallow, deep copy 就是字面上的意思
        dataset_args = copy.deepcopy(dataset_spec['args'])
        # wrapper 裡面的args =>　inp_size, scale_max, augment, sample_q, 所以上面做完，dataset_args 這個dict就會有 wrapper 裡面的資料
        dataset_args.update(args)
        # 最後再把第一次make的 class ImageFolder object 也丟進來, 所以等等要丟進class sr_implicit_downsampled 裡面的就會有:
        # wrapper 參數: inp_size, scale_max, augment, sample_q + class ImageFolder object

    else:
        dataset_args = dataset_spec['args']
    temp = datasets
    dataset = datasets[dataset_spec['name']](**dataset_args)
    # datasets這個dict裡面裝了來自 image_folder.py, wrappers.py 裡面的各式各樣的class
    """
    第一次make:
    用小寫名子 (ex: "image-folder") 去叫出對應的class並用他來製造object (但並非最後的) 
    至於那個** 看一下notion   
    
    第二次make:
    用小寫名子 (ex: "sr_implicit_downsampled") 去叫出對應的class並用他來製造object => 最後最完整的dataset, 等等丟到dataloader裡面
      
    """

    return dataset
