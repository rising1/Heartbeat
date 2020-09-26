    document.getElementById("prediction").innerHTML = "awaited";

    var dragHandler = function(evt){
        evt.preventDefault();
    };

    var dropHandler = function(evt){
        evt.preventDefault();
        var files = evt.originalEvent.dataTransfer.files;

        var formData = new FormData();
        formData.append("file2upload", files[0]);

        var req = {
            url: "/uploader",
            //url: "/test",
            // url: "/sendfile",
            method: "post",
            processData: false,
            contentType: false,
            data: formData
        };
        console.log('Posted');

        // $.ajax(req)
         $.ajax(req)
            .done(function(data){
                $('#prediction').text(data).show();
            } );

         event.preventDefault();
    };

    var dropHandlerSet = {
        dragover: dragHandler,
        drop: dropHandler
    };

    $(".droparea").on(dropHandlerSet);


