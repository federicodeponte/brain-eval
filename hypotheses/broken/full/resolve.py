import ast, json, re
def _s(node):
    """Render a string-ish AST node; f-string placeholders become '{}'."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _s(node.left), _s(node.right)
        return None if a is None or b is None else a + b
    return None
def _name_parts(tree, name, depth):
    """Strings a script puts into variable `name` (assign / += / append / extend), in source order."""
    parts = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            v = n.value.body if isinstance(n.value, ast.IfExp) else n.value
            if isinstance(v, (ast.List, ast.Tuple)):
                parts = [x for x in (_str_of(tree, el, depth + 1) for el in v.elts) if x]
            else:
                sv = _str_of(tree, v, depth + 1)
                if sv is not None:
                    parts = [sv]
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == name:
            v = n.value
            if isinstance(v, (ast.List, ast.Tuple)):
                parts += [x for x in (_str_of(tree, el, depth + 1) for el in v.elts) if x]
            else:
                sv = _str_of(tree, v, depth + 1)
                if sv:
                    parts.append(sv)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
              and n.func.value.id == name and n.args):
            if n.func.attr == "append":
                sv = _str_of(tree, n.args[0], depth + 1)
                if sv:
                    parts.append(sv)
            elif n.func.attr == "extend" and isinstance(n.args[0], (ast.List, ast.Tuple)):
                parts += [x for x in (_str_of(tree, el, depth + 1) for el in n.args[0].elts) if x]
    return parts


def _str_of(tree, node, depth):
    if depth > 4 or node is None:
        return None
    if isinstance(node, ast.IfExp):
        node = node.body
    sv = _s(node)
    if sv is not None:
        return sv
    if isinstance(node, ast.Name):
        p = _name_parts(tree, node.id, depth)
        return "; ".join(p) if p else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" and node.args:
        sep = _s(node.func.value) or "; "
        a = node.args[0]
        if isinstance(a, ast.Name):
            p = _name_parts(tree, a.id, depth)
        elif isinstance(a, (ast.List, ast.Tuple)):
            p = [x for x in (_str_of(tree, el, depth + 1) for el in a.elts) if x]
        else:
            p = []
        return sep.join(p) if p else None
    return None


def resolve_var(traj):
    """Last command that json.dumps a dict with data_issues=<Name>: resolve Name from that script's assignments."""
    best = None
    for l in open(traj):
        e = json.loads(l); it = e.get("item", {})
        if e.get("type") != "item.completed" or it.get("type") != "command_execution":
            continue
        cmd = it.get("command") or ""
        m = re.search(r"<<\s*'?(\w+)'?\n(.*)\n\1", cmd, re.S)
        if not m:
            continue
        code = m.group(2).replace('\\"', '"').replace("\\$", "$").replace("\\`", "`")
        try:
            tree = ast.parse(code)
        except SyntaxError:
            try:  # undo bash quote splicing: python x!=d['k'] is written inside bash -lc "..." as x"'!=d['"'k']
                tree = ast.parse(code.replace("\"'", "").replace("'\"", ""))
            except SyntaxError:
                continue
        target = None
        for n in ast.walk(tree):
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if isinstance(k, ast.Constant) and k.value == "data_issues":
                        target = v
        if target is None:
            continue
        if isinstance(target, ast.IfExp):  # "; ".join(issues) if issues else "none"
            target = target.body
        val = _str_of(tree, target, 0)
        if val:
            best = val
    return best
