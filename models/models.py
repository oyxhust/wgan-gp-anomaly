
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'wgan-gp':
        from .wgan_gp_model import WGAN_GP_Model
        model = WGAN_GP_Model()
    elif opt.model == 'test':
        from .test_model import TEST_Model
        model = TEST_Model()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
