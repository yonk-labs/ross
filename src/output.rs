use serde_json::Value;

pub fn json_out(results: &[Value], single: bool) -> String {
    if single {
        if let Some(v) = results.first() {
            return format!("{}\n", serde_json::to_string_pretty(v).unwrap());
        }
    }
    format!("{}\n", serde_json::to_string_pretty(results).unwrap())
}

fn field_lines(v: &Value) -> String {
    let mut out = String::new();
    let path = v["path"].as_str().unwrap_or("(unknown)");
    let mut push = |k: &str, val: String| {
        out.push_str(&format!("  {k}: {val}\n"));
    };
    if let Some(e) = v["error"].as_str() {
        push("error", e.to_string());
    }
    for k in ["type", "bytes", "mtime", "sha256"] {
        match &v[k] {
            Value::String(s) => push(k, s.clone()),
            Value::Number(n) => push(k, n.to_string()),
            _ => {}
        }
    }
    if let Some(tags) = v["tags"].as_array() {
        let t: Vec<String> = tags
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect();
        if !t.is_empty() {
            push("tags", t.join(", "));
        }
    }
    for k in ["description", "summary", "model", "duration_ms"] {
        if let Some(s) = v[k].as_str() {
            if !s.is_empty() {
                push(k, s.to_string());
            }
        } else if let Some(n) = v[k].as_u64() {
            push(k, format!("{n}ms"));
        }
    }
    if !v["metadata"].is_null() {
        out.push_str("  metadata:\n");
        if let Ok(p) = serde_json::to_string_pretty(&v["metadata"]) {
            for line in p.lines() {
                out.push_str(&format!("    {line}\n"));
            }
        }
    }
    out.push_str(&format!("  path: {path}\n"));
    out
}

pub fn md_out(results: &[Value]) -> String {
    let mut out = String::new();
    for v in results {
        out.push_str(&format!("## {}\n", v["path"].as_str().unwrap_or("(unknown)")));
        out.push_str(&field_lines(v));
        out.push('\n');
    }
    out
}

pub fn text_out(results: &[Value]) -> String {
    let mut out = Vec::new();
    for v in results {
        let block = format!(
            "== {} ==\n{}",
            v["path"].as_str().unwrap_or("(unknown)"),
            field_lines(v)
        );
        out.push(block.trim_end().to_string());
    }
    format!("{}\n", out.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn entry() -> Value {
        json!({
            "path": "/a/b.png", "type": "image", "bytes": 10, "mtime": 5,
            "sha256": "abc", "tags": ["cat", "dog"], "description": "a cat",
            "summary": "cat sits", "model": "m", "duration_ms": 7,
            "metadata": {"width": 4, "height": 2}
        })
    }

    #[test]
    fn json_single_is_object() {
        let s = json_out(&[entry()], true);
        let v: Value = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(v["tags"][0], "cat");
    }

    #[test]
    fn json_multi_is_array() {
        let s = json_out(&[entry(), entry()], false);
        let v: Value = serde_json::from_str(s.trim()).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2);
    }

    #[test]
    fn md_renders_sections() {
        let s = md_out(&[entry()]);
        assert!(s.contains("## /a/b.png"));
        assert!(s.contains("tags: cat, dog"));
        assert!(s.contains("description: a cat"));
    }

    #[test]
    fn text_renders_blocks() {
        let s = text_out(&[entry()]);
        assert!(s.contains("== /a/b.png =="));
        assert!(s.contains("model: m"));
        assert!(s.contains("duration_ms: 7ms"));
    }

    #[test]
    fn error_entry_minimal() {
        let s = text_out(&[json!({"path": "/x", "error": "boom"})]);
        assert!(s.contains("error: boom"));
    }
}
