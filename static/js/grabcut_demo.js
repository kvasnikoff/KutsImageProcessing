$(document).ready(function () {
  console.log("doc ready");
  all_rows = {}; // id: objc
  appending_container = "appending_container";

  project_name_1 = "burger";
  let obj1 = makeGrabCutRow(
    appending_container,
    "../static/images/burger/_original/ss.png",
    project_name_1
  );
  let obj1_id = obj1.id;
  all_rows[obj1_id] = obj1;
});
