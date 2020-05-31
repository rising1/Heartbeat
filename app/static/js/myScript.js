    var dragHandler = function(evt){
        evt.preventDefault();
    };

    var dropHandler = function(evt){
        evt.preventDefault();
        var files = evt.originalEvent.dataTransfer.files;
        console.log(files[0]);
    };

    var dropHandlerSet = {
        dragover: dragHandler,
        drop: dropHandler
    };
    if (typeof $(".droparea") != "undefined") {
        alert("GOT THERE");
    }
    $(".droparea").on(dropHandlerSet);