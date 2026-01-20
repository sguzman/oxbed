use unicode_normalization::UnicodeNormalization;

pub fn normalize(
  input: &str
) -> String {
  let mut normalized =
    String::with_capacity(input.len());
  let mut last_was_space = false;
  let mut newline_count = 0;
  for ch in input.nfkc() {
    match ch {
      | '\r' => continue,
      | '\n' => {
        if newline_count < 2 {
          normalized.push('\n');
        }
        newline_count += 1;
        last_was_space = true;
      }
      | c if c.is_whitespace() => {
        if newline_count > 0 {
          newline_count = 0;
        }
        if !last_was_space {
          normalized.push(' ');
          last_was_space = true;
        }
      }
      | other => {
        newline_count = 0;
        normalized.push(other);
        last_was_space = false;
      }
    }
  }
  normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
  use super::normalize;

  #[test]
  fn normalize_collapses_whitespace_and_nfkc()
   {
    let raw = "\u{FB01}   bar baz";
    let normalized = normalize(raw);
    assert_eq!(
      normalized,
      "fi bar baz"
    );
  }
}
