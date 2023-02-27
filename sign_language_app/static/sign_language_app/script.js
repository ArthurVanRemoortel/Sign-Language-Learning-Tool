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

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function getCsrfToken(){
    return getCookie('csrftoken')
}