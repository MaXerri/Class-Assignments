let string_of_list ?(open_delim = "[") ?(close_delim = "]") ?(sep = "; ")
    string_of_elt lst =
  let len = List.length lst in
  let open Buffer in
  (* As a rough lower bound assume that each element takes a minimum of 3
     characters to represent including a separator, e.g., ["v, "]. The buffer
     will grow as needed, so it's okay if that estimate is low. *)
  let buf = create (3 * len) in
  add_string buf open_delim;
  List.iteri
    (fun i v ->
      add_string buf (string_of_elt v);
      if i < len - 1 then add_string buf sep)
    lst;
  add_string buf close_delim;
  contents buf

let string_of_bindings key_to_string value_to_string lst =
  let string_of_binding (k, v) =
    Printf.sprintf "%s: %s" (key_to_string k) (value_to_string v)
  in
  string_of_list ~open_delim:"{" ~close_delim:"}" ~sep:", " string_of_binding
    lst
