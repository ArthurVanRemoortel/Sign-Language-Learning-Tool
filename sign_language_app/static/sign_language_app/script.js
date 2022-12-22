function makeSelect2($inputRef, queryUrl, processResults, dataFunction, formatFunction, formatSelectionFunction, placeholder) {
    $inputRef.select2({
        ajax: {
            delay: 500,
            url: queryUrl,
            dataType: "json",
            type: "GET",
            data: function (params) {
                return {
                    name_like: params.term,
                };
            },
            processResults: processResults,
            cache: false
        },
        minimumInputLength: 2,
        templateResult: formatFunction,
        templateSelection: formatSelectionFunction,
        placeholder: placeholder,
        allowClear: true
    });
}

