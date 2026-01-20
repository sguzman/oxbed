use unicode_normalization::UnicodeNormalization;

pub fn normalize(
  input: &str
) -> String {
  let mut normalized =
    String::with_capacity(input.len());
  let mut last_was_space = false;
  for ch in input.nfc() {
    match ch {
      | '\r' => continue,
      | '\n' => {
        if normalized.ends_with('\n') {
          continue;
        }
        normalized.push('\n');
        last_was_space = true;
      }
      | c if c.is_whitespace() => {
        if !last_was_space {
          normalized.push(' ');
          last_was_space = true;
        }
      }
      | other => {
        normalized.push(other);
        last_was_space = false;
      }
    }
  }
  normalized.trim().to_string()
}
