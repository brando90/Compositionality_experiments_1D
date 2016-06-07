function train_obj = get_train_min_obj(info,dummy1)
train_obj = nanmin([info.train(:).objective]);


