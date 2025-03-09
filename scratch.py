for k, v in col_customization_dict.items():
    col_type = map_custom_dtype(v.dtype)
    if v.is_fk:
        attrs[v.new_name] = Column(col_type, ForeignKey(f'{v.new_name}_lookup.id'))
    else:
        attrs[v.new_name] = Column(col_type)

return type(table_name, (Base,), attrs)
