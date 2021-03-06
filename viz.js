function hex(i) {
    var str = Number(i).toString(16);
    return str.length == 1 ? "0" + str : str;
}

function showEntropy() {
    $('.tok').each(function (index, el) {
	el = $(el);
	var entropy = el.attr('data-entropy-relative');
	var val = Math.floor(255 - parseFloat(entropy) * 255);
	var color = "#FF" + hex(val) + hex(val);
	el.css('background-color', color);
    });
}

function showScore() {
    $('.tok').each(function (index, el) {
	el = $(el);
	var score = el.attr('data-score-relative');
	var val = Math.floor((1.0 - parseFloat(score) * 0.5) * 255);
	var color = "#" + hex(val) + hex(val) + "FF";
	el.css('background-color', color);
    });
}

function clearHighlighting() {
    $('.tok').each(function (index, el) {
	el = $(el);
	el.css('background-color', '');
    });
}

function indicateChanges() {
    $('.changed-tok').each(function (index, el) {
	el = $(el);
	el.css('font-weight', 'bold');
    });
}

function hideChanges() {
    $('.changed-tok').each(function (index, el) {
	el = $(el);
	el.css('font-weight', '');
    });
}

function showPopup(e) {
    var el = $(e.target);
    var score = el.attr('data-score');
    var replacements = JSON.parse(el.attr('data-replacements'));
    $(".box").remove();
    var html = "<div class='box'>Top prediction: " + replacements.join('/') + "<br />";
    html += "Score: " + Number(score).toFixed(3) + "<hr /><table class='graph-holder'><tr>";
    for (var k = 1; k <= 3; k++) {
	var options = JSON.parse(el.attr("data-options" + k));
	var entropy = el.attr('data-entropy' + k);
	if (options == null) continue;
	html += "<td><table>";
	if (k == 1) html += "No topic: <br />";
	if (k == 2) html += "Raw: <br />";
	if (k == 3) html += "Constrained: <br />";
	var max = 0.0;
	for (var i = 0; i < options.length; i++) {
	    var p = parseFloat(options[i][1]);
	    if (p > max) max = p;
	}
	for (var i = 0; i < options.length; i++) {
	    html += "<tr><td>";
	    html += options[i][0];
	    html += "</td><td><div class='bar' style='width: " + (options[i][1] / max) * 40 + "px'>&nbsp;</div>";
	    html += "</td></tr>";
	}
	html += "</table>Entropy: " + Number(entropy).toFixed(3) + "</td>";
    }
    html += "</tr></table></div>";
    $("body").append(html);
}

$(function () {
    showScore();
    $("#changes").change(function (e) {
	if ($("#changes").prop('checked')) {
	    indicateChanges();
	} else {
	    hideChanges();
	}
    });
    $("#highlighting").change(function (e) {
	if ($("#highlighting").val() == 'Entropy') {
	    showEntropy();
	} else if ($("#highlighting").val() == 'Score') {
	    showScore();
	} else {
	    clearHighlighting();
	}
    });
    $(".tok").on("dblclick", function (e) {
	showPopup(e);
    });
})
