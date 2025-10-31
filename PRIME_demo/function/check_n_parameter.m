function check_n_parameter()
    paths = { ...
        ['./network/result/model_epoch.pt'], ...
        ['./network/result/train_time_file'] ...
    };

    cellfun(@(path) delete_if_exists(path), paths);
end


function delete_if_exists(filePath)
    if exist(filePath, 'file') == 2
        delete(filePath);
    end
end