<h2 align="center">
<a href="https://www.onyx.app/">

  
![logotype](https://github.com/onyx-dot-app/onyx/blob/logo/OnyxLogoCropped.jpg?raw=true)

</a>

</h2>

# Onyx Docs

This repo generates the docs and setup guide for [Onyx](https://github.com/onyx-dot-app/onyx) found at [https://docs.onyx.app/](https://docs.onyx.app/).

It uses Mintlify which is a low-code documentation generation tool.

More info on Mintlify found [here](https://mintlify.com/).

To make changes, check out `docs.json`. 

### Set up Mintlify

Install the [Mintlify CLI](https://www.npmjs.com/package/mintlify) to preview the documentation changes locally.

To install, use the following command (requires node >= v19.0.0)

```
npm i -g mintlify
```

Run the following command at the root of your documentation (where docs.json is)

```
mintlify dev
```

### Publishing Changes

Changes are automatically deployed to production after merging to main.

### Troubleshooting

- Mintlify dev isn't running - Run `mintlify install` to re-install dependencies.
- Page loads as a 404 - Make sure you are running in a folder with `docs.json`
- Mintlify Docs - https://mintlify.com/docs/introduction
