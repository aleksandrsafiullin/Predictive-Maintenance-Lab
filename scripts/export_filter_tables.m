function export_filter_tables(input_file, output_dir)
% Export actual MATLAB tables/timetables; no MCOS guessing or outcome inference.
arguments
    input_file (1,1) string
    output_dir (1,1) string
end
if ~isfolder(output_dir), mkdir(output_dir); end
loaded = load(input_file);
manifest.schema_version = 'filter_table_export_v1';
manifest.source_file = char(input_file);
manifest.matlab_version = version;
manifest.exported_tables = {};
manifest.unsupported_variables = {};
fields = fieldnames(loaded);
for n = 1:numel(fields)
    name = fields{n}; value = loaded.(name);
    if istable(value) || istimetable(value)
        info.variable = name;
        info.original_class = class(value);
        if istimetable(value), value = timetable2table(value); end
        info.columns = value.Properties.VariableNames;
        info.units = value.Properties.VariableUnits;
        info.descriptions = value.Properties.VariableDescriptions;
        info.rows = height(value);
        info.file = [name '.csv'];
        writetable(value, fullfile(output_dir, info.file));
        % Reload and compare every numeric column against the actual table.
        restored = readtable(fullfile(output_dir, info.file), 'VariableNamingRule', 'preserve');
        info.numeric_roundtrip_verified = true;
        for c = 1:width(value)
            if isnumeric(value{:,c})
                a = value{:,c}; b = restored{:,c};
                if ~isequal(size(a),size(b)) || any(abs(a-b) > 1e-10 .* max(1,abs(a)), 'all')
                    error('Numeric roundtrip mismatch in %s/%s', name, info.columns{c});
                end
            end
        end
        manifest.exported_tables{end+1} = info;
    else
        manifest.unsupported_variables{end+1} = struct('variable',name,'class',class(value));
    end
end
manifest.origin_mapping_verified = false;
manifest.time_units_verified = false;
manifest.status = 'requires_origin_endpoint_and_unit_verification';
fid = fopen(fullfile(output_dir, 'manifest.json'), 'w');
cleanup = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(manifest, PrettyPrint=true));
end
